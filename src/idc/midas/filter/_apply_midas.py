import argparse
from typing import List

import numpy as np
import torch

from wai.logging import LOGGING_WARNING
from idc.api import DepthData, DEVICES, DEVICE_AUTO, make_list, flatten_list, DepthInformation
from seppl.io import Filter

MODEL_SMALL = "MiDaS_small"
MODEL_HYBRID = "DPT_Hybrid"
MODEL_LARGE = "DPT_Large"
MODELS = [
    MODEL_SMALL,
    MODEL_HYBRID,
    MODEL_LARGE,
]


class ApplyMidas(Filter):
    """
    Applies MiDaS to the image and overrides the depth information.
    """

    def __init__(self, model: str = None, device: str = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the filter.

        :param model: the MiDaS model to use
        :type model: str
        :param device: the device to run the model on
        :type device: str
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)
        self.model = model
        self.device = device
        self._midas = None
        self._device = None
        self._transform = None

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "apply-midas"

    def description(self) -> str:
        """
        Returns a description of the filter.

        :return: the description
        :rtype: str
        """
        return "Applies MiDaS to the image and overrides the depth information. For more information see: https://pytorch.org/hub/intelisl_midas_v2/"

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [DepthData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [DepthData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-m", "--model", choices=MODELS, help="The MiDaS model to use.", default=MODEL_SMALL, required=False)
        parser.add_argument("-d", "--device", choices=DEVICES, help="The device to run the model on.", default=DEVICE_AUTO, required=False)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.model = ns.model
        self.device = ns.device

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.model is None:
            self.model = MODEL_SMALL
        if self.device is None:
            self.device = DEVICE_AUTO

        # device
        if self.device == DEVICE_AUTO:
            self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self._device = torch.device(self.device)

        # model
        self.logger().info("Loading model: %s" % self.model)
        self._midas = torch.hub.load("intel-isl/MiDaS", self.model)
        self._midas.to(self._device)
        self._midas.eval()

        # transform
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model == MODEL_LARGE or self.model == MODEL_HYBRID:
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform

    def _do_process(self, data):
        """
        Processes the data record(s).

        :param data: the record(s) to process
        :return: the potentially updated record(s)
        """
        result = []
        for item in make_list(data):
            img = np.asarray(item.image)
            input_batch = self._transform(img).to(self._device)

            with torch.no_grad():
                prediction = self._midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            depth_info = prediction.cpu().numpy()

            new_item = item.duplicate(annotation=DepthInformation(depth_info))
            result.append(new_item)

        return flatten_list(result)

    def finalize(self):
        """
        Finishes the processing, e.g., for closing files or databases.
        """
        super().finalize()
        self._transform = None
        self._device = None
        self._midas = None
