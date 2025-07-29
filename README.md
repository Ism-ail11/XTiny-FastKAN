# XTiny-FastKAN: Tifinagh Air-Writing Recognition using TinyML

![License](https://img.shields.io/badge/license-MIT-green)
![Framework](https://img.shields.io/badge/TensorFlow-2.x-blue)
![TinyML](https://img.shields.io/badge/TinyML-Ready-success)

## 📌 Overview

This project introduces **XTiny-FastKAN**, a lightweight, interpretable, and highly efficient neural network for **gesture-based air handwriting recognition** of the **Tifinagh script**, targeting real-time inference on low-power **TinyML microcontrollers**.

This work includes:
- A novel **Tifinagh IMU-based air-writing dataset**
- A rasterization pipeline to transform motion trajectories into color-encoded images
- A customized version of **FastKAN** optimized for TinyML
- Robust **data augmentation** and preprocessing strategies
- Reproducible training, evaluation, and explainability via saliency maps

---

## 🚀 Features

✅ First air-writing dataset for **Tifinagh characters**  
✅ Compact and fast model (~35 KB, < 0.05ms latency)  
✅ Compatible with **TensorFlow Lite Micro / Edge Impulse**  
✅ Includes **data augmentation**, training scripts, and **confusion matrix tools**  
✅ Built-in **XAI module** (saliency visualization)

---

## 📂 Repository Structure

