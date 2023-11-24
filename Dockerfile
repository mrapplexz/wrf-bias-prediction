FROM pytorch/pytorch


#RUN conda install -c conda-forge wrf-python=1.3.4.1
COPY . /home
WORKDIR /home
RUN pip install -r requirements.txt



#CMD ["python", "experiments/constantBaseline/main.py"]
