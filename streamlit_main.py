import streamlit as st
from predictionOnImage import return_prediction
from PIL import Image
from matplotlib import pyplot as plt
import time
import os
from keras.models import load_model

# thiết lập tiêu đề 
st.title("Distracted Driver Detection")

fig = plt.figure()

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a model', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)
def main():
    st.sidebar.title('Thành viên nhóm')
    st.sidebar.subheader('VĂN HIẾU - HỮU DỰ - TIẾN ĐẠT')
    # lấy path hình ảnh được chọn
    file_uploaded = st.file_uploader("Chọn File", type=["png", "jpg", "jpeg"])
    # thiết lập button phân loại
    class_btn = st.button("Classify")

    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write('image',image)

    if class_btn:
        if file_uploaded is None:
            st.write("tải lại ảnh")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                st.write('image',image)
                model = load_model(filename)
                predictions = return_prediction(image,model)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                # st.pyplot(fig)


if __name__ == '__main__':
    main()
