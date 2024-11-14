FROM colomoto/colomoto-docker:2024-11-01

USER root
RUN conda install -y pydruglogics

USER user
