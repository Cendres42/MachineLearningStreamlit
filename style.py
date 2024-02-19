import base64
import streamlit as st

# Charger une image pour le fond
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)




def styled_write(content):
    st.write(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 10px; border-radius: 10px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )
    
def styled_write2(content):
    st.write(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 10px; border-radius: 10px;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )
    
def styled_h2(content):
    st.write(
        f"""
        <div style="background-color: rgba(255, 255, 255, 0.7); padding: 10px; border-radius: 10px;color:darkred;">
            {content}
        </div>
        """,
        unsafe_allow_html=True
    )