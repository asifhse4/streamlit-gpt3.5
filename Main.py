import streamlit as st
from model import GeneralModel
from PIL import Image


def app():

    # Creating an object of prediction service
    pred = GeneralModel()

    image = Image.open('listing_image.png')

    st.image(image, caption='50 Candle Ter SW Calgary, AB')
    
    # https://www.ojohome.ca/calgary-ab/50-candle-ter-sw-calgary-ab-t2w-6g7/pid_yealetr3cm/
    
    api_key = st.sidebar.text_input("APIkey", type="password")
    
    # Using the streamlit cache
    @st.cache
    def process_prompt(input):
        return pred.model_prediction(input=input.strip() , api_key=api_key)

    if api_key:

        # Setting up the Title
        st.title("Ask me a question about this home")

        # st.write("---")

        s_example = "Is there a park nearby?"
        input = st.text_area(
            "Use the example below or input your own text in English",
            value=s_example,
            max_chars=300,
            height=100,
        )

        if st.button("Submit"):
            with st.spinner(text="In progress"):
                report_text = process_prompt(input)
                st.markdown(report_text)
    else:
        st.error("🔑 Please enter API Key")
