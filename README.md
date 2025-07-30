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
├── data/ # Raw and processed Tifinagh stroke data

├── rasterization/ # Converts strokes into color raster images

├── augmentation/ # Augmentation scripts (rotation, noise, warping

├── model/ # XTiny-FastKAN model and architecture

├── training/ # Training and evaluation scripts

├── explainability/ # XAI module (saliency maps)

├── tflite/ # Exported .tflite and quantized models

├── results/ # Evaluation metrics, plots, confusion matrix

├── Deployment/ Deploy our XTiny-FastKAN into edge device

├── notebooks/ # Jupyter notebooks for visualization and testing

└── README.md

## 📄 License

This project is licensed under the terms of the **MIT License**.  
You are free to use, modify, distribute, and sell this software, provided that you include the original copyright and license notice.


📬 Contact
For questions or collaborations, please contact:

Ismail Lamaakal

 Multidisciplinary Faculty of Nador, Mohammed Premier University, Oujda, Morocco
 
📧 ismail.lamaakal@ieee.org
