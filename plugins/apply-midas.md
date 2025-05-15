# apply-midas

* accepts: idc.api.DepthData
* generates: idc.api.DepthData

Applies MiDaS to the image and overrides the depth information. For more information see: https://pytorch.org/hub/intelisl_midas_v2/

```
usage: apply-midas [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                   [-N LOGGER_NAME] [--skip]
                   [-m {MiDaS_small,DPT_Hybrid,DPT_Large}]
                   [-d {auto,cpu,cuda}]

Applies MiDaS to the image and overrides the depth information. For more
information see: https://pytorch.org/hub/intelisl_midas_v2/

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -m {MiDaS_small,DPT_Hybrid,DPT_Large}, --model {MiDaS_small,DPT_Hybrid,DPT_Large}
                        The MiDaS model to use. (default: MiDaS_small)
  -d {auto,cpu,cuda}, --device {auto,cpu,cuda}
                        The device to run the model on. (default: auto)
```
