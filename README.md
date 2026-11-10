# RODIC InfraAI — Hosted POC

AI-assisted infrastructure visual inspection POC for the
RODIC InfraAI Innovation Challenge 2026.

## Application

Streamlit-based multimodal infrastructure inspection application.

The application combines:

- Domain-trained computer vision
- CODEBRIM defect classification
- SDNET2018 crack classification
- Gemini multimodal visual evidence
- Evidence arbitration
- Human engineering validation boundary

## Responsible AI Boundary

The system organizes and reconciles visual evidence.

It does **not**:

- determine structural safety
- provide engineering certification
- make final engineering dispositions
- treat model confidence as calibrated probability
- average confidence values across independent evidence sources

Qualified human inspection remains responsible for engineering
significance and final disposition.

## Credential Configuration

The Gemini API credential is not stored in this repository.

Configure the following Streamlit secret:

`GEMINI_API_KEY`

## Deployment

This repository is intended for Streamlit Community Cloud deployment.
