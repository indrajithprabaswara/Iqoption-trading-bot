# IQ Option Trading Bot

This repository contains experimental trading scripts for the IQ Option platform. The code relies on the unofficial `iqoptionapi` library.

## Credentials

`iqo12_fixed.py` expects the IQ Option username and password to be provided via environment variables:

```
export IQOPTION_USERNAME="your_email@example.com"
export IQOPTION_PASSWORD="your_password"
```

Optional: set `IQOPTION_TEST_MODE=1` to run the script without connecting to IQ Option.

## Accuracy Expectations

The scripts attempt to train predictive models on historical data but there is **no guarantee** that they will achieve a particular accuracy or make profitable trades. Real trading results depend on many factors outside the scope of this project.

Use this repository for educational purposes only and do not risk funds you cannot afford to lose.

