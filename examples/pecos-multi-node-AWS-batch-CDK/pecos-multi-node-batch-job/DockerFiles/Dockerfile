FROM ubuntu:20.04


ENV TZ=America/New_York \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install tzdata

RUN apt-get update && apt-get install sudo
RUN sudo apt-get update && sudo apt-get install python3 python3-dev mpich -y
RUN sudo apt-get update && sudo apt-get install -y build-essential git python3 python3-distutils python3-venv
RUN sudo apt-get install -y libopenblas-dev
RUN sudo apt-get install -y python3-pip
RUN sudo apt-get install -y awscli
RUN python3 -m pip install supervisor
RUN python3 -m pip install mpi4py
RUN python3 -m pip install libpecos


# DockerfileBuild adds this user
RUN groupadd -r ecs-user && useradd --no-log-init -r -g ecs-user ecs-user
RUN echo "ecs-user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

RUN apt-get update -y && apt-get install -y openssl openssh-client openssh-server iproute2


# Setup user env
ENV USER ecs-user
ENV HOME /home/$USER
RUN echo $HOME
RUN chown -R ecs-user:ecs-user /home


# Enable password-less SSH
ENV SSHDIR $HOME/.ssh
RUN mkdir -p ${SSHDIR} \
&& touch ${SSHDIR}/sshd_config \
&& ssh-keygen -t rsa -f ${SSHDIR}/ssh_host_rsa_key -N '' \
&& cp ${SSHDIR}/ssh_host_rsa_key.pub ${SSHDIR}/authorized_keys \
&& cp ${SSHDIR}/ssh_host_rsa_key ${SSHDIR}/id_rsa \
&& echo " IdentityFile ${SSHDIR}/id_rsa" >> ${SSHDIR}/config \
&& echo " StrictHostKeyChecking no" >> ${SSHDIR}/config \
&& echo " UserKnownHostsFile /dev/null" >> ${SSHDIR}/config \
&& echo " Port 2022" >> ${SSHDIR}/config \
&& echo 'Port 2022' >> ${SSHDIR}/sshd_config \
&& echo 'UsePrivilegeSeparation no' >> ${SSHDIR}/sshd_config \
&& echo "HostKey ${SSHDIR}/ssh_host_rsa_key" >> ${SSHDIR}/sshd_config \ && echo "PidFile ${SSHDIR}/sshd.pid" >> ${SSHDIR}/sshd_config \
&& chmod -R 600 ${SSHDIR}/* \
&& chown -R ${USER}:${USER} ${SSHDIR}/
RUN eval `ssh-agent -s` && ssh-add ${SSHDIR}/id_rsa
EXPOSE 22

# Add script
USER ecs-user
ADD ./CorePecosImageBuildScripts/supervisord.conf /etc/supervisor/supervisord.conf
ADD ./CorePecosImageBuildScripts/mpi-run.sh /supervised-scripts/mpi-run.sh
RUN sudo chmod 755 /supervised-scripts/mpi-run.sh
ADD ./CorePecosImageBuildScripts/entry-point.sh /batch-runtime-scripts/entry-point.sh
RUN sudo chmod 0755 /batch-runtime-scripts/entry-point.sh
