# Logging

🤗 Transformers has a centralized logging system, so that you can setup the verbosity of the library easily.

Currently the default verbosity of the library is `WARNING`.

To change the level of verbosity, just use one of the direct setters. For instance, here is how to change the verbosity to the INFO level.

```
import transformers

transformers.logging.set_verbosity_info()
```

You can also use the environment variable `TRANSFORMERS_VERBOSITY` to override the default verbosity. You can set it to one of the following: `debug`, `info`, `warning`, `error`, `critical`. For example:

```
TRANSFORMERS_VERBOSITY=error ./myprogram.py
```

Additionally, some `warnings` can be disabled by setting the environment variable `TRANSFORMERS_NO_ADVISORY_WARNINGS` to a true value, like _1_. This will disable any warning that is logged using `logger.warning_advice`. For example:

```
TRANSFORMERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

Here is an example of how to use the same logger as the library in your own module or script:

```
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
```

All the methods of this logging module are documented below, the main ones are [logging.get\_verbosity()](/docs/transformers/v4.34.0/en/main_classes/logging#transformers.utils.logging.get_verbosity) to get the current level of verbosity in the logger and [logging.set\_verbosity()](/docs/transformers/v4.34.0/en/main_classes/logging#transformers.utils.logging.set_verbosity) to set the verbosity to the level of your choice. In order (from the least verbose to the most verbose), those levels (with their corresponding int values in parenthesis) are:

-   `transformers.logging.CRITICAL` or `transformers.logging.FATAL` (int value, 50): only report the most critical errors.
-   `transformers.logging.ERROR` (int value, 40): only report errors.
-   `transformers.logging.WARNING` or `transformers.logging.WARN` (int value, 30): only reports error and warnings. This the default level used by the library.
-   `transformers.logging.INFO` (int value, 20): reports error, warnings and basic information.
-   `transformers.logging.DEBUG` (int value, 10): report all information.

By default, `tqdm` progress bars will be displayed during model download. [logging.disable\_progress\_bar()](/docs/transformers/v4.34.0/en/main_classes/logging#transformers.utils.logging.disable_progress_bar) and [logging.enable\_progress\_bar()](/docs/transformers/v4.34.0/en/main_classes/logging#transformers.utils.logging.enable_progress_bar) can be used to suppress or unsuppress this behavior.

## Base setters

#### transformers.utils.logging.set\_verbosity\_error

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L195)

( )

Set the verbosity to the `ERROR` level.

#### transformers.utils.logging.set\_verbosity\_warning

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L185)

( )

Set the verbosity to the `WARNING` level.

#### transformers.utils.logging.set\_verbosity\_info

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L180)

( )

Set the verbosity to the `INFO` level.

#### transformers.utils.logging.set\_verbosity\_debug

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L190)

( )

Set the verbosity to the `DEBUG` level.

## Other functions

#### transformers.utils.logging.get\_verbosity

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L138)

( ) → `int`

Return the current level for the 🤗 Transformers’s root logger as an int.

🤗 Transformers has following logging levels:

-   50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
-   40: `transformers.logging.ERROR`
-   30: `transformers.logging.WARNING` or `transformers.logging.WARN`
-   20: `transformers.logging.INFO`
-   10: `transformers.logging.DEBUG`

#### transformers.utils.logging.set\_verbosity

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L161)

( verbosity: int )

Parameters

-   **verbosity** (`int`) — Logging level, e.g., one of:
    
    -   `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
    -   `transformers.logging.ERROR`
    -   `transformers.logging.WARNING` or `transformers.logging.WARN`
    -   `transformers.logging.INFO`
    -   `transformers.logging.DEBUG`
    

Set the verbosity level for the 🤗 Transformers’s root logger.

#### transformers.utils.logging.get\_logger

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L124)

( name: typing.Optional\[str\] = None )

Return a logger with the specified name.

This function is not supposed to be directly accessed unless you are writing a custom transformers module.

#### transformers.utils.logging.enable\_default\_handler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L209)

( )

Enable the default handler of the HuggingFace Transformers’s root logger.

#### transformers.utils.logging.disable\_default\_handler

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L200)

( )

Disable the default handler of the HuggingFace Transformers’s root logger.

#### transformers.utils.logging.enable\_explicit\_format

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L255)

( )

Enable explicit formatting for every HuggingFace Transformers’s logger. The explicit formatter is as follows:

```
    [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
```

All handlers currently bound to the root logger are affected by this method.

#### transformers.utils.logging.reset\_format

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L270)

( )

Resets the formatting for HuggingFace Transformers’s loggers.

All handlers currently bound to the root logger are affected by this method.

#### transformers.utils.logging.enable\_progress\_bar

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L361)

( )

Enable tqdm progress bar.

#### transformers.utils.logging.disable\_progress\_bar

[< source \>](https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/utils/logging.py#L368)

( )

Disable tqdm progress bar.