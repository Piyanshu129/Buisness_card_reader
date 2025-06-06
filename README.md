# 📇 OCR Business Card Reader with EasyOCR & Streamlit

This is a powerful Streamlit web application that uses **EasyOCR** to extract and auto-categorize data from business card images. It supports **multi-card upload**, intelligent data extraction, automatic grouping by company domain, and exports to CSV, PDF, and vCard formats.

---

## 🚀 Features

- 🔍 Optical Character Recognition (OCR) with EasyOCR
- 🧠 Intelligent categorization (Name, Email, Phone, Website, Address, Pincode)
- 📸 Upload multiple business card images
- 📦 Export extracted data to:
  - CSV
  - PDF
  - vCard (.vcf)
- 🏢 Auto-grouping by company domain
- 🛢 Optional MySQL database integration

---

## 📦 Tech Stack

- **Python**
- **Streamlit**
- **EasyOCR**
- **Pandas**, **re**, **SQLAlchemy**
- **FPDF** for PDF generation
- **MySQL** (optional)

## Application link
https://buisnesscardreader-8elqxekhzwldmagly9qzya.streamlit.app/
---


## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/Piyanshu129/Buisness_card_reader.git
cd Buisness_card_reader

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate



pip install -r requirements.txt


streamlit run buisnesscard.py

or

docker build -t ocr-card-reader .
docker run -p 8501:8501 ocr-card-reader

