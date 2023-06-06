import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags
import plotly.graph_objects as go
import cv2
import numpy as np
from configs import IMAGE_SIZE, TEST_CLASS, TRAIN_CLASS, BACKEND, SUPPORT_PATH
from utils import auth_load_bucket, download_support_set, get_support_images_and_labels, process_queries
from utils import get_table_experiment, show_experiment_metrics, get_embedding, display_model_graph, inference, zipdir
import time
import os
import zipfile
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title='Batik Motifs')

os.environ['MLFLOW_TRACKING_URI'] = st.secrets['MLFLOW_TRACKING_URI']
os.environ['MLFLOW_TRACKING_USERNAME'] = st.secrets['MLFLOW_TRACKING_USERNAME']
os.environ['MLFLOW_TRACKING_PASSWORD'] = st.secrets['MLFLOW_TRACKING_PASSWORD']

selected = option_menu(
    None, # "Main Menu",
    ["Home", "Recognition", "Labeling"],
    menu_icon='list',
    icons=['house', 'binoculars', 'pencil-square'],
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {"padding": "1!important", "background-color": "#F0F2F6"},
        "icon": {"color": "#31333F", "font-size": "16px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#FFFFFF"},
        "nav-link-selected": {"background-color": "#FF4B4B"}
    }
)

if selected == 'Home':
    st.markdown("# Batik Motifs Recognition and Labeling Hub :rocket:")
    st.write('---')
    with st.container():
        st.markdown("### :partying_face: Welcome to Batik Motifs Analysis")
        p1 = """
            Yo, welcome to our Batik Motif Recognition and Labeling project. This dope interactive platform is all about 
            recognizing batik motifs using baddas machine learning technique, like prototypical networks. It can spot new 
            batik motifs that haven't been trained before with just a few samples, no need to train the model again.
        """
        p2 = """
            We also got some sick analytics features, like uncertainty analysis, to check how the model deals with data dirft 
            and potential prediction errors. It's all about keeping tabs on the model's performance and staying on top of any 
            funcky stuff that might pop up.
        """
        p3 = """
            And we've also thrown in this dope semi-automatic labeing feature that allows for some sweet human-in-the-loop action. 
            But here's the deal, we ain't updating the model based on new data or data drift for a few reasons (storage constraints, 
            expert domain validation, ect). However, this feature still gives user the chance to contribute their expertise to the 
            labeling process, strictly for their own spesific needs.
        """
        p4 = """
            Are you a Batik enthusiast, a researcher, or just curious about the fascinating world of Batik? Well, this platform offer 
            you an exciting and informative experience. Get ready to explore and immerse yourself in the captivating world of Batik!
        """
        # col1, col2 = st.columns(2)
        st.write(p1)
        st.write(p2)
        st.write(p3)

    with st.container():
        st.markdown("### :woozy_face: Model Testing Benchmark")
        st.write(
            """
            Benchmarking results for novel batik motif classes from the best model under various conditions.
            [Notebook](https://dagshub.com/afhabibieee/fsl-batik/src/main/src/pytorch/test.ipynb) 
            [Experiments](https://dagshub.com/afhabibieee/fsl-batik/experiments/#/)
            """
        )
        df = pd.read_csv('best-model-benchmarks.csv', header=[0,1], index_col=0)
        st.table(df)
        st.info(
            """
            The above information can serve as reference for determining the number of classes and the number of images per class when 
            using the model for recognition and labeling.
            """
        )
    st.write('---')
    st.write(p4)


elif selected == 'Recognition':
    # st.session_state
    st.markdown("# #1 Let's find out what the batik pattern is! :male_mage:")

    tab1, tab2, tab3 = st.tabs([':file_folder: Data', ':robot_face: Model', ':mag: Inference'])

    # if st.session_state is None:
    #     st.session_state = {}

    if 'button1' not in st.session_state:
        st.session_state.button1 = False
    if 'button2' not in st.session_state:
        st.session_state.button2 = False
    if 'button3' not in st.session_state:
        st.session_state.button3 = False

    state_vars = [
        'support_images', 'support_labels', 'label_decoded',
        'query', 'queries_tensors', 
        'model', 'cams',
        'most_conf', 'predicted_classes',
        'least_confs', 'margin_confs', 'ratio_confs', 'entorpies',
    ]

    for state_var in state_vars:
        if state_var not in st.session_state:
            st.session_state[state_var] = None

    with tab1:
        with st.container():
            st.subheader(":open_file_folder: Support Set")
            st.write("Enter the batik motifs you're trying to recognize!")
            selected_batik = st_tags(
                label='',
                text='Press enter to add more',
                value=TEST_CLASS,
                suggestions=TRAIN_CLASS,
                maxtags=20,
                key='1'
            )

            with st.expander('Other available batik motifs'):
                st.write(
                    f"""
                    The tags above represent some batik motifs that are different from those used in training.
                    You can add some batik motifs that were used during training.
                    \n`Train Class`: {TRAIN_CLASS}
                    \n`Val/Test Class`: {TEST_CLASS}
                    """
                )

            st.write("Select the number of supporting images for each class that you want/have")
            N_SHOT = st.slider(label='', min_value=1, max_value=20, step=1, label_visibility='collapsed')

            for batik in selected_batik:
                if batik not in TRAIN_CLASS+TEST_CLASS:
                    st.write(f"Upload {N_SHOT} example images for the supporting set of {batik}")
                    
                    uploaded_support_imgs = st.file_uploader('Upload a few images', type=['png', 'jpg'], accept_multiple_files=True)
                    
                    for i, uploaded_file in enumerate(uploaded_support_imgs, start=1):
                        if uploaded_file is not None:
                            byte_data = uploaded_file.read()
                            folder = os.path.join(SUPPORT_PATH, batik)
                            filename = f'{i}.png'

                            os.makedirs(folder, exist_ok=True)

                            with open(os.path.join(folder, filename), 'wb') as f:
                                f.write(byte_data)
                            
                            st.success(f'Image {i} saved to disk')
    
        with st.container():
            st.subheader(":open_file_folder: Query Set")
            uploaded_query = st.file_uploader("Upload an image", type=['png', 'jpg'])
    
        if uploaded_query is not None:
            query = np.asarray(bytearray(uploaded_query.read()), dtype=np.uint8)
            query = cv2.imdecode(query, 1)
            query = query[:,:,::-1]
            query = cv2.resize(query, (IMAGE_SIZE, IMAGE_SIZE))
            st.session_state['query'] = query

        with st.container():
            if st.button('Process support and query sets'):
                st.session_state.button1 = True
                progress_bar = st.progress(0)
                notification = st.empty()

                b2_id = st.secrets['B2_KEY_ID']
                b2_app_key = st.secrets['B2_APPLICATION_KEY']
                b2_bucket = auth_load_bucket(b2_id, b2_app_key)
                download_support_set(b2_bucket, selected_batik)
                support_images, support_labels, label_decoded = get_support_images_and_labels(selected_batik, N_SHOT)
                if uploaded_query is not None:   
                    list_query = [query,]
                    queries_tensors = process_queries(list_query)

                st.session_state['support_images'] = support_images
                st.session_state['support_labels'] = support_labels
                st.session_state['label_decoded'] = label_decoded
                st.session_state['queries_tensors'] = queries_tensors

                for i in range(100):
                    progress_bar.progress(i+1)
                    time.sleep(0.01)
                progress_bar.empty()
                notification.success(f'Download and preprocessing have been complete!')

    with tab2:
        with st.container():
            st.markdown('### :card_index_dividers: Model-Registry')
            st.write("Here's a collection of experiments that includes the parameters used and the metrics obtained")
            st.markdown('##### Table experiments')
            exp_data = get_table_experiment()
            st.dataframe(data=exp_data)
        
        with st.container():
            st.markdown('### :dart: Select Model')
            
            st.write("Let's choose the best model!")
            run_id_option = [f'{name} : {id}' for (id, name) in zip(exp_data.index.to_list(), exp_data['params.backbone_name'].to_list())]
            run_id_option = st.selectbox("##### Select the desired model", (None, *run_id_option))
            if run_id_option is not None:
                st.markdown('##### Metrics loss and accuracy')
                run_id = run_id_option.split(' : ')[-1]
                backbone_name = run_id_option.split(' : ')[0]
                col1, col2 = st.columns(2)
                exp_metrics = show_experiment_metrics(run_id)

                # container plot metrics
                with st.container():
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['train_loss'], mode='lines', name='train loss'))
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['val_loss'], mode='lines', name='validation loss'))
                        fig.update_layout(
                            xaxis_title='Epochs',
                            yaxis_title='Losses'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['train_acc'], mode='lines', name='train accuracy'))
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['val_acc'], mode='lines', name='validation accuracy'))
                        fig.update_layout(
                            xaxis_title='Epochs',
                            yaxis_title='Accuracies'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # container model
                with st.container():
                    backend = st.selectbox('##### Select the desired backend', (BACKEND))
                    if st.button('Get model artifact'):
                        st.session_state.button2 = True
                        progress_bar = st.progress(0)
                        notification = st.empty()
                        model = get_embedding(run_id, backend)
                        st.session_state['model'] = model

                        for i in range(100):
                            progress_bar.progress(i+1)
                            time.sleep(0.01)
                        progress_bar.empty()
                        notification.success(f'Download artifact model complete!')

                        display_model_graph(model, run_id)
                        with open(f'model-registry/{run_id}/model/model.png', 'rb') as img_file:
                            bytes_img = img_file.read()
                        st.download_button(
                            label='Download model graph',
                            data=bytes_img,
                            file_name='graph model.png',
                            mime='image/png'
                        )
    
    with tab3:
        st.markdown("### :bomb: Prediction")
        col1, col2 = st.columns(2)
        if st.session_state['query'] is not None:
            with col1:
                fig, ax = plt.subplots()
                ax.imshow(st.session_state['query'])
                ax.axis('off')
                st.pyplot(fig)
                st.markdown("<p style='text-align: center;'>Image origin</p>", unsafe_allow_html=True)

                if st.button('Inference!', use_container_width=True):
                    st.session_state.button3 = True
                    progress_bar = st.progress(0)
                    notification = st.empty()

                    most_conf, predicted_classes, least_confs, margin_confs, ratio_confs, entorpies, cams = inference(
                        st.session_state['support_images'], 
                        st.session_state['support_labels'], 
                        st.session_state['queries_tensors'], 
                        st.session_state['model'], 
                        st.session_state['label_decoded'], 
                        mode='recognition'
                    )
                    st.session_state['most_conf'] = most_conf
                    st.session_state['predicted_classes'] = predicted_classes
                    st.session_state['least_confs'] = least_confs
                    st.session_state['margin_confs'] = margin_confs
                    st.session_state['ratio_confs'] = ratio_confs
                    st.session_state['entorpies'] = entorpies
                    st.session_state['cams'] = cams

                    for i in range(100):
                        progress_bar.progress(i+1)
                        time.sleep(0.01)
                    progress_bar.empty()
                    notification.success('Inference complete :)')

            if st.session_state['cams'] is not None:    
                with col2:
                    fig, ax = plt.subplots()
                    ax.imshow(st.session_state['query'])
                    ax.imshow(st.session_state['cams'], cmap='jet', alpha=0.5)
                    ax.axis('off')
                    st.pyplot(fig)
                    st.markdown("<p style='text-align: center;'>Class activation map image by Grad-CAM</p>", unsafe_allow_html=True)

                    pred_output = {
                        'Predicted class': st.session_state['predicted_classes'],
                        'Confidence score': st.session_state['most_conf']
                    }
                    pred_output = pd.DataFrame(pred_output)
                    pred_output.set_index('Predicted class', inplace=True)
                    st.dataframe(pred_output, use_container_width=True)

        if st.session_state.button3 and st.session_state['least_confs'] is not None:
            with st.container():
                st.markdown('### :bar_chart: Advanced Analytics')

                with st.expander('Uncertainty analysis :sparkles:'):
                    uncertainty_metrics = {
                        'Least confidence': st.session_state['least_confs'][0].item(),
                        'Margin of cofidence': st.session_state['margin_confs'][0].item(),
                        'Ratio of confidence': st.session_state['ratio_confs'][0].item(),
                        'Entropy': st.session_state['entorpies'][0].item()
                    }

                    thresh = st.number_input('Insert a threshlold', min_value=0.0, max_value=1.0, value=0.6)

                    cols = st.columns(4)
                    calc_drift_count = 0

                    with st.container():
                        for i, (label, value) in enumerate(uncertainty_metrics.items()):
                            delta = thresh - value
                            cols[i].metric(label=label, value=round(value, 4), delta=round(delta, 2))
                            calc_drift_count = calc_drift_count + 1 if value > thresh else calc_drift_count
                
                    if calc_drift_count > 0:
                        st.info('We found data drift and potential prediction errors!', icon="ℹ️")

        with st.container():
            if st.button('Reset session state', use_container_width=True):
                st.session_state.clear()

elif selected == 'Labeling':
    # st.session_state.clear()

    # if st.session_state is None:
    #     st.session_state = {}

    st.markdown("# #2 Human-in-the-loop semi-automated labeling :female_mage:")

    tab1, tab2, tab3 = st.tabs([':file_folder: Data', ':robot_face: Model', ':mag: Inference & Labeling'])


    if 'button1' not in st.session_state:
        st.session_state.button1 = False
    if 'button2' not in st.session_state:
        st.session_state.button2 = False
    if 'button3' not in st.session_state:
        st.session_state.button3 = False
    if 'finish_labeling' not in st.session_state:
        st.session_state.finish_labeling = False
    
    state_vars = [
        'selected_batik',
        'support_images', 'support_labels', 'label_decoded',
        'queries_unresize',
        'queries', 'queries_tensors', 
        'model',
        'most_conf', 'predicted_classes',
        'least_confs', 'margin_confs', 'ratio_confs', 'entorpies',
        'byteszip'
    ]

    for state_var in state_vars:
        if state_var not in st.session_state:
            st.session_state[state_var] = None

    with tab1:
        with st.container():
            st.subheader(":open_file_folder: Support Set")
            st.write("Enter the batik motifs you're trying to recognize!")
            selected_batik = st_tags(
                label='',
                text='Press enter to add more',
                value=TEST_CLASS,
                suggestions=TRAIN_CLASS,
                maxtags=20,
                key='1'
            )
            st.session_state['selected_batik'] = selected_batik

            with st.expander('Other available batik motifs'):
                st.write(
                    f"""
                    The tags above represent some batik motifs that are different from those used in training.
                    You can add some batik motifs that were used during training.
                    \n`Train Class`: {TRAIN_CLASS}
                    \n`Val/Test Class`: {TEST_CLASS}
                    """
                )

            st.write("Select the number of supporting images for each class that you want/have")
            N_SHOT = st.slider(label='', min_value=1, max_value=20, step=1, label_visibility='collapsed')

            for batik in selected_batik:
                if batik not in TRAIN_CLASS+TEST_CLASS:
                    st.write(f"Upload {N_SHOT} example images for the supporting set of {batik}")
                    
                    uploaded_support_imgs = st.file_uploader('Upload a few images', type=['png', 'jpg'], accept_multiple_files=True)
                    
                    for i, uploaded_file in enumerate(uploaded_support_imgs, start=1):
                        if uploaded_file is not None:
                            byte_data = uploaded_file.read()
                            folder = os.path.join(SUPPORT_PATH, batik)
                            filename = f'{i}.png'

                            os.makedirs(folder, exist_ok=True)

                            with open(os.path.join(folder, filename), 'wb') as f:
                                f.write(byte_data)
                            
                            st.success(f'Image {i} saved to disk')
    
        with st.container():
            st.subheader(":open_file_folder: Query Set")
            uploaded_queries = st.file_uploader("Upload any image", type=['png', 'jpg'], accept_multiple_files=True)

        queries = []
        queries_unresize = []
        if uploaded_queries is not None:
            for uploaded_query in uploaded_queries:
                query = np.asarray(bytearray(uploaded_query.read()), dtype=np.uint8)
                query = cv2.imdecode(query, 1)
                query = query[:,:,::-1]
                queries_unresize.append(query)
                query = cv2.resize(query, (IMAGE_SIZE, IMAGE_SIZE))
                queries.append(query)
        
        st.session_state['queries'] = queries
        st.session_state['queries_unresize'] = queries_unresize

        with st.container():
            if st.button('Process support and query sets'):
                st.session_state.button1 = True
                progress_bar = st.progress(0)
                notification = st.empty()

                b2_id = st.secrets['B2_KEY_ID']
                b2_app_key = st.secrets['B2_APPLICATION_KEY']
                b2_bucket = auth_load_bucket(b2_id, b2_app_key)
                download_support_set(b2_bucket, selected_batik)
                support_images, support_labels, label_decoded = get_support_images_and_labels(selected_batik, N_SHOT)
                if uploaded_queries is not None:   
                    queries_tensors = process_queries(st.session_state['queries'])

                st.session_state['support_images'] = support_images
                st.session_state['support_labels'] = support_labels
                st.session_state['label_decoded'] = label_decoded
                st.session_state['queries_tensors'] = queries_tensors

                for i in range(100):
                    progress_bar.progress(i+1)
                    time.sleep(0.01)
                progress_bar.empty()
                notification.success(f'Download and preprocessing have been complete!')

    with tab2:
        with st.container():
            st.markdown('### :card_index_dividers: Model-Registry')
            st.write("Here's a collection of experiments that includes the parameters used and the metrics obtained")
            st.markdown('##### Table experiments')
            exp_data = get_table_experiment()
            st.dataframe(data=exp_data)
        
        with st.container():
            st.markdown('### :dart: Select Model')
            
            st.write("Let's choose the best model!")
            run_id_option = [f'{name} : {id}' for (id, name) in zip(exp_data.index.to_list(), exp_data['params.backbone_name'].to_list())]
            run_id_option = st.selectbox("##### Select the desired model", (None, *run_id_option))
            if run_id_option is not None:
                st.markdown('##### Metrics loss and accuracy')
                run_id = run_id_option.split(' : ')[-1]
                backbone_name = run_id_option.split(' : ')[0]
                col1, col2 = st.columns(2)
                exp_metrics = show_experiment_metrics(run_id)

                # container plot metrics
                with st.container():
                    with col1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['train_loss'], mode='lines', name='train loss'))
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['val_loss'], mode='lines', name='validation loss'))
                        fig.update_layout(
                            xaxis_title='Epochs',
                            yaxis_title='Losses'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['train_acc'], mode='lines', name='train accuracy'))
                        fig.add_trace(go.Scatter(x=exp_metrics.index, y=exp_metrics['val_acc'], mode='lines', name='validation accuracy'))
                        fig.update_layout(
                            xaxis_title='Epochs',
                            yaxis_title='Accuracies'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # container model
                with st.container():
                    backend = st.selectbox('##### Select the desired backend', (BACKEND))
                    if st.button('Get model artifact'):
                        st.session_state.button2 = True
                        progress_bar = st.progress(0)
                        notification = st.empty()
                        model = get_embedding(run_id, backend)
                        st.session_state['model'] = model

                        for i in range(100):
                            progress_bar.progress(i+1)
                            time.sleep(0.01)
                        progress_bar.empty()
                        notification.success(f'Download artifact model complete!')

                        display_model_graph(model, run_id)
                        with open(f'model-registry/{run_id}/model/model.png', 'rb') as img_file:
                            bytes_img = img_file.read()
                        st.download_button(
                            label='Download model graph',
                            data=bytes_img,
                            file_name='graph model.png',
                            mime='image/png'
                        )

    with tab3:
        st.markdown("### :bomb: Prediction")
        thresh = st.number_input('Insert a threshlold', min_value=0.0, max_value=1.0, value=0.6)
        if st.button('Inference!', use_container_width=True):
            st.session_state.button3 = True
            progress_bar = st.progress(0)
            notification = st.empty()

            most_conf, predicted_classes, least_confs, margin_confs, ratio_confs, entorpies, _ = inference(
                st.session_state['support_images'], 
                st.session_state['support_labels'], 
                st.session_state['queries_tensors'], 
                st.session_state['model'], 
                st.session_state['label_decoded'], 
                mode='labeling'
            )
            st.session_state['most_conf'] = most_conf
            st.session_state['predicted_classes'] = predicted_classes
            st.session_state['least_confs'] = least_confs
            st.session_state['margin_confs'] = margin_confs
            st.session_state['ratio_confs'] = ratio_confs
            st.session_state['entorpies'] = entorpies

            for i in range(100):
                progress_bar.progress(i+1)
                time.sleep(0.01)
            progress_bar.empty()
            notification.success('Inference complete :)')

        if st.session_state['queries'] is not None and st.session_state.button3:
            total_drift = 0 
            for i, query in enumerate(st.session_state['queries']):
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        fig, ax = plt.subplots()
                        ax.imshow(query)
                        ax.axis('off')
                        st.pyplot(fig)
                        # st.markdown("<p style='text-align: center;'>Image no.{}</p>".format(i+1), unsafe_allow_html=True)
                        
                    with col2:
                        pred_output = {
                            'Predicted class': st.session_state['predicted_classes'][i],
                            'Confidence score': st.session_state['most_conf'][i],
                            'Least confidence': st.session_state['least_confs'][i].item(),
                            'Margin of cofidence': st.session_state['margin_confs'][i].item(),
                            'Ratio of confidence': st.session_state['ratio_confs'][i].item(),
                            'Entropy': st.session_state['entorpies'][i].item()
                        }
                        pred_output = pd.DataFrame(pred_output, index=[0]).T
                        pred_output.rename(columns={0: 'Values'}, inplace=True)
                        st.dataframe(pred_output, use_container_width=True)
                        
                        drift_metric_count = 0
                        for calc in pred_output.loc[['Least confidence', 'Margin of cofidence', 'Ratio of confidence', 'Entropy'], 'Values']:
                            if calc > thresh:
                                drift_metric_count += 1
                        
                        selected_class = st.selectbox(
                            "Predicted/labeled:",
                            st.session_state['selected_batik'],
                            st.session_state['selected_batik'].index(st.session_state['predicted_classes'][i]),
                            key=f'select_class_{i}'
                        )
                        st.session_state[f'labeled_class_{i}'] = selected_class

                    info_placeholder = st.empty()
                    if drift_metric_count > 0:
                        info_placeholder.info('We found data drift and potential prediction errors! Time for re-labeling.', icon="❎")    
                        total_drift += 1
                    else:
                        info_placeholder.info('This image labeling is taken care of by the model.', icon="✅")

            st.info(f"We need to re-label {total_drift} out of the {len(st.session_state['queries'])} input images", icon="ℹ️")
                            
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if st.button('Finish up the labeling', use_container_width=True):
                    st.session_state.finish_labeling = True
                    folder = 'labeling'
                    if st.session_state['queries_unresize'] is not None:
                        for i, uploaded_query in enumerate(st.session_state['queries_unresize']):
                            class_path = os.path.join(folder, st.session_state[f'labeled_class_{i}'])
                            os.makedirs(class_path, exist_ok=True)
                            cv2.imwrite(os.path.join(class_path, f'img{i}.jpg'), uploaded_query)
                    
                        # prepare the zip file
                        zipf = zipfile.ZipFile('labeling.zip', 'w', zipfile.ZIP_DEFLATED)
                        zipdir(folder, zipf)
                        zipf.close()

                        # read the zip file as bytes
                        with open('labeling.zip', 'rb') as f:
                            byteszip = f.read()
                            st.session_state['byteszip'] = byteszip
            
            with col2:
                if st.session_state.finish_labeling:
                    st.download_button(
                        label='Download labeled images',
                        data=st.session_state['byteszip'],
                        file_name='labeling.zip',
                        mime='application/zip',
                        use_container_width=True
                    )

        with st.container():
                if st.button('Reset session state', use_container_width=True):
                    st.session_state.clear()