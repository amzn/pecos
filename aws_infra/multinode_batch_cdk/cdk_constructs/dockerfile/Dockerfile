FROM amazonlinux:2


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Fix vulnerability
RUN yum update -y gnupg2
RUN yum -y updateinfo && yum -y --security update

# Install dependencies
RUN yum install -y awscli sudo git
RUN yum install -y openssl openssh-clients openssh-server iproute
RUN yum install -y python3 python3-devel python3-distutils python3-venv
RUN yum groupinstall -y 'Development Tools'
RUN amazon-linux-extras install -y epel
RUN yum install -y openblas-devel

# Install MPI
RUN yum install -y http://mirror.centos.org/centos/7/os/x86_64/Packages/mpich-3.2-3.2-2.el7.x86_64.rpm
RUN yum install -y http://mirror.centos.org/centos/7/os/x86_64/Packages/mpich-3.2-devel-3.2-2.el7.x86_64.rpm
RUN touch /etc/profile.d/mpich.sh
RUN echo 'export PATH=/usr/lib64/mpich-3.2/bin/:$PATH' | tee /etc/profile.d/mpich.sh
ENV PATH="/usr/lib64/mpich-3.2/bin/:${PATH}"

RUN python3 -m pip install supervisor
RUN python3 -m pip install mpi4py

# Install pecos from local source
ADD README.md /pecos-source/README.md
ADD Makefile /pecos-source/Makefile
ADD setup.py /pecos-source/setup.py
ADD pecos /pecos-source/pecos/
RUN cd /pecos-source && make clean && make libpecos

# Setup user
RUN groupadd -r amazon && useradd --no-log-init -r -g amazon ecs-user
RUN echo "ecs-user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
ENV USER ecs-user
ENV HOME /home/$USER
RUN echo $HOME
RUN mkdir -p $HOME && chown -R ecs-user:amazon $HOME

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
&& chown -R ${USER}:amazon ${SSHDIR}/
RUN eval `ssh-agent -s` && ssh-add ${SSHDIR}/id_rsa
EXPOSE 22

# Add script
USER ecs-user
ARG SCRIPTS_DIR=aws_infra/multinode_batch_cdk/cdk_constructs/dockerfile/scripts
ADD ${SCRIPTS_DIR}/supervisord.conf /etc/supervisor/supervisord.conf
ADD ${SCRIPTS_DIR}/dist-run.sh /supervised-scripts/dist-run.sh
RUN sudo chmod 755 /supervised-scripts/dist-run.sh
ADD ${SCRIPTS_DIR}/entry-point.sh /batch-runtime-scripts/entry-point.sh
RUN sudo chmod 755 /batch-runtime-scripts/entry-point.sh
