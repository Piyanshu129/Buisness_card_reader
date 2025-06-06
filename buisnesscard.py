import asyncio
import os
import easyocr
import streamlit as st
from PIL import Image
import numpy as np
import re
import pandas as pd
from sqlalchemy import create_engine
from fpdf import FPDF

# Fix Mac event loop
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

st.set_page_config(layout="wide")
reader = easyocr.Reader(['en'], gpu=False)

st.title("📇 Business Card Extractor with Auto-Grouping")

uploaded_files = st.file_uploader(
    "Upload one or more business card images",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# ---------- Helper Functions ----------

def clean_email(email):
    if '@' not in email:
        return email
    local, domain = email.split('@', 1)
    if '.' not in domain:
        common_tlds = ['com', 'in', 'org', 'net', 'co', 'us', 'edu', 'gov']
        for tld in common_tlds:
            if domain.endswith(tld):
                domain = domain[:-len(tld)] + '.' + tld
                break
    return local + '@' + domain

def normalize_website(raw):
    raw = raw.lower().strip()
    raw = re.sub(r"\s+", "", raw)
    raw = re.sub(r"[^a-z0-9.-]", "", raw)
    if not raw.startswith("www."):
        raw = "www." + raw.lstrip("www")
    if not re.search(r"\.\w{2,}$", raw):
        raw += ".com"
    return raw

def extract_emails_and_websites(text_lines):
    emails = []
    websites = []

    email_pattern = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.\w{2,}\b")
    website_pattern = re.compile(r"www\.[a-zA-Z0-9\-\s\.]+\.\w{2,}")

    for line in text_lines:
        line_lower = line.lower()
        cleaned_line = re.sub(r'(?i)(email|supportemail|emailid|web|website)\s*[:\-]?', '', line_lower)

        found_emails = email_pattern.findall(cleaned_line)
        emails.extend(found_emails)

        if '@' in cleaned_line and not found_emails:
            cleaned_line = cleaned_line.replace(' ', '')
            if '@' in cleaned_line:
                emails.append(cleaned_line)

        found_sites = website_pattern.findall(line_lower)
        if not found_sites and 'www' in line_lower:
            parts = re.split(r'[|,]', line_lower)
            for part in parts:
                if 'www' in part:
                    websites.append(normalize_website(part))
        else:
            websites.extend([normalize_website(site) for site in found_sites])

    emails = list(set([clean_email(e) for e in emails]))
    websites = list(set(websites))
    return emails, websites

def extract_text_with_confidence(image):
    results = reader.readtext(np.array(image), detail=1)
    extracted = []
    for _, text, conf in results:
        if text.strip():
            extracted.append({"text": text.strip(), "confidence": round(conf * 100, 2)})
    return extracted

def categorize_fields(lines):
    data = {
        "Card Holder": [],
        "Email": [],
        "Phone": [],
        "Website": [],
        "Pincode": [],
        "Address": [],
    }

    address_keywords = ['road', 'rd', 'st', 'sector', 'nagar', 'building', 'complex', 'village',
                        'area', 'hall', 'block', 'lane', 'layout', 'mumbai', 'vashi', 'opp', 'city']

    emails, websites = extract_emails_and_websites(lines)
    data["Email"].extend(emails)
    data["Website"].extend(websites)

    def extract_full_address(lines):
        address_block = []
        address_started = False

        for line in lines:
            if not address_started and re.search(r'\bAdd\b[:：]?', line, re.IGNORECASE):
                address_started = True
                address_block.append(re.sub(r'\bAdd\b[:：]?', '', line, flags=re.IGNORECASE).strip())
            elif address_started:
                if re.search(r'\b(Contact|Phone|Email|Support|Web|www|@|[0-9]{10}|Pincode)\b', line, re.IGNORECASE):
                    break
                address_block.append(line.strip())

        return address_block

    address_block = extract_full_address(lines)
    if address_block:
        data["Address"] = address_block
    else:
        for line in lines:
            line_lower = line.lower()
            if '@' not in line and any(keyword in line_lower for keyword in address_keywords):
                data["Address"].append(line)

    for line in lines:
        line_lower = line.lower()
        line_cleaned = re.sub(r'(?i)(Mob|contact|support)\s*[:\-]?', '', line_lower)
        phones = re.findall(r"\b\d{10}\b", line_cleaned)
        if phones:
            data["Phone"].extend(phones)

        pincodes = re.findall(r"\b\d{6}\b", line)
        if pincodes:
            data["Pincode"].extend(pincodes)

    for line in lines:
        line_clean = line.strip()
        if (not any(char.isdigit() for char in line_clean) and
                len(line_clean.split()) <= 3 and
                line_clean == line_clean.title() and
                not any(keyword in line_clean.lower() for keyword in address_keywords) and
                not re.search(r"[:,]", line_clean)):
            data["Card Holder"].append(line_clean)
            break

    return data

def create_df(data_dict):
    max_len = max(len(v) for v in data_dict.values())
    for key in data_dict:
        while len(data_dict[key]) < max_len:
            data_dict[key].append(None)
    return pd.DataFrame(data_dict)

def generate_vcf(data_dict):
    lines = []
    for i in range(len(data_dict["Card Holder"])):
        name = data_dict["Card Holder"][i] or "Unknown"
        phone = data_dict["Phone"][i] if i < len(data_dict["Phone"]) else ""
        email = data_dict["Email"][i] if i < len(data_dict["Email"]) else ""
        address = data_dict["Address"][i] if i < len(data_dict["Address"]) else ""

        lines.append(f"""BEGIN:VCARD
VERSION:3.0
FN:{name}
TEL;TYPE=CELL:{phone}
EMAIL:{email}
ADR;TYPE=HOME:;;{address}
END:VCARD
""")
    return "\n".join(lines)

def extract_company_identifier(row):
    if pd.notna(row["Email"]):
        domain_match = re.search(r"@([a-zA-Z0-9.-]+)", str(row["Email"]))
        if domain_match:
            return domain_match.group(1).lower()
    if pd.notna(row["Website"]):
        website = str(row["Website"]).lower().replace("www.", "")
        return website.split('/')[0]
    return "Unknown"

def generate_pdf_from_df(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    col_names = df.columns.tolist()

    for i in range(len(df)):
        row = df.iloc[i]
        for col in col_names:
            pdf.cell(200, 10, txt=f"{col}: {row[col]}", ln=1)
        pdf.ln(5)

    return pdf.output(dest='S').encode("latin-1")

# ---------- Processing Loop ----------
all_dfs = []
if uploaded_files:
    for file in uploaded_files:
        st.markdown("---")
        st.image(Image.open(file), caption=f"📇 {file.name}", use_column_width=True)

        with st.spinner(f"Extracting text from {file.name}..."):
            image = Image.open(file)
            extracted_lines = extract_text_with_confidence(image)
            lines = [item["text"] for item in extracted_lines]

        st.markdown("#### 📄 Extracted Text:")
        for item in extracted_lines:
            st.write(f"**{item['text']}** — 🧠 `{item['confidence']}%`")

        with st.spinner("Categorizing..."):
            data = categorize_fields(lines)
            df = create_df(data)
            all_dfs.append(df)

        st.markdown("#### 📋 Categorized Information:")
        st.json(data)

# ---------- Combined Actions ----------
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    def fill_card_holder(df):
        mask = df.drop(columns=['Card Holder']).notnull().any(axis=1)
        df.loc[:, 'Card Holder'] = df['Card Holder'].replace('', np.nan)
        df.loc[mask, 'Card Holder'] = df.loc[mask, 'Card Holder'].ffill()
        return df

    combined_df = fill_card_holder(combined_df)
    combined_df['Company'] = combined_df.apply(extract_company_identifier, axis=1)

    st.markdown("## 📊 Final Combined Table")
    edited_df = st.data_editor(combined_df, use_container_width=True, num_rows="dynamic")

    search_query = st.text_input("🔍 Search by Card Holder")
    if search_query:
        filtered_df = combined_df[combined_df['Card Holder'].str.contains(search_query, case=False, na=False)]
        if not filtered_df.empty:
            card_holder_name = filtered_df['Card Holder'].iloc[0]
            combined_address = ', '.join(
                filtered_df['Address'].dropna().astype(str).unique()
            )
            st.markdown(f"### 🧍 Card Holder: `{card_holder_name}`")
            st.markdown("#### 📍 Combined Address:")
            st.success(combined_address)
            st.dataframe(filtered_df)
        else:
            st.warning("No matching Card Holder found.")
    else:
        st.dataframe(edited_df)

    st.markdown("## 🏢 Grouped by Company")
    for company, group in combined_df.groupby("Company"):
        st.subheader(f"🏢 {company}")
        st.dataframe(group.reset_index(drop=True))

    csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button("📄 Download Combined CSV", data=csv, file_name="combined_cards.csv", mime="text/csv")

    combined_dict = edited_df.to_dict(orient='list')
    vcf_data = generate_vcf(combined_dict)
    st.download_button("📅 Export All as vCard (.vcf)", vcf_data, file_name="contacts.vcf", mime="text/vcard")

    pdf_data = generate_pdf_from_df(edited_df)
    st.download_button("📄 Export All as PDF", data=pdf_data, file_name="contacts.pdf", mime="application/pdf")

    db_user = "root"
    db_password = "saini_dev79"
    db_host = "localhost"
    db_port = "3306"
    db_name = "business_cards"

    db_url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)

    if st.button("📅 Save All to Database"):
        try:
            edited_df.to_sql('card_holder_company', con=engine, if_exists='append', index=False)
            st.success("✅ All data saved to SQL database successfully.")
        except Exception as e:
            st.error(f"❌ Database Error: {e}")
