# SCMARL irrigation coordination — performance evaluation code

This repository contains Python source files for the paper's performance-evaluation workflows and shared simulation utilities.

## Excluded from this code-only release

Simulation inputs, trained model weights/checkpoints, generated results, logs, virtual environments, and editor settings are not included. Obtain the required research inputs and model checkpoints separately before running the workflows.

## Partial source snapshot

This code-only release omits `common/environments/richards_equation_scheduler.py`. Simulation workflows that import this module require an authorized copy of it before they can run.

## LLM API configuration

For DashScope-compatible calls, set `DASHSCOPE_API_KEY` and `DASHSCOPE_API_URL` in the runtime environment. No API key or third-party API endpoint is stored in the source. Ollama calls use a local endpoint configured in the relevant scripts.

## Scope

The source tree is organized by experiment family. Please refer to the manuscript for the experimental protocol and interpretation of reported results.
