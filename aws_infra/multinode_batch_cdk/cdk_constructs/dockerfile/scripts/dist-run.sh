#!/bin/bash
set -x


BASENAME="${0##*/}"
log () {
  echo "${BASENAME} - ${1}"
}
HOST_FILE_PATH="/tmp/hostfile"
NODE_IP_FILE_PREFIX="/tmp/node_ip_"
NODE_IP_FILE_PATH="${NODE_IP_FILE_PREFIX}${AWS_BATCH_JOB_ID}"
AWS_BATCH_EXIT_CODE_FILE="/tmp/batch-exit-code"



usage () {
  if [ "${#@}" -ne 0 ]; then
    log "* ${*}"
    log
  fi
  cat <<ENDUSAGE
Usage:
export AWS_BATCH_JOB_NODE_INDEX=0
export AWS_BATCH_JOB_NUM_NODES=10
export AWS_BATCH_JOB_MAIN_NODE_INDEX=0
export AWS_BATCH_JOB_ID=string
./dist-run.sh
ENDUSAGE

  error_exit
}

# Standard function to print an error and exit with a failing return code
error_exit () {
  log "${BASENAME} - ${1}" >&2
  log "${2:-1}" > $AWS_BATCH_EXIT_CODE_FILE
  kill  $(cat /tmp/supervisord.pid)
}

# Set child by default switch to main if on main node container
NODE_TYPE="child"
if [ "${AWS_BATCH_JOB_MAIN_NODE_INDEX}" == "${AWS_BATCH_JOB_NODE_INDEX}" ]; then
  log "Running synchronize as the main node"
  NODE_TYPE="main"
fi

install_deps() {
  if [[ ! -v BATCH_BOOTSTRAP ]]; then
    echo "BATCH_BOOTSTRAP is not set. continuing"
  else
    $BATCH_BOOTSTRAP
  fi
}

# wait for all nodes to report
wait_for_nodes () {
  log "Running as master node"

  touch $NODE_IP_FILE_PATH
  ip=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)

  if [ -x "$(command -v nvidia-smi)" ] ; then
      NUM_GPUS=$(ls -l /dev/nvidia[0-9] | wc -l)
      availablecores=$NUM_GPUS
  else
      availablecores=$(nproc)
  fi

  log "master details -> $ip:$availablecores"
  echo "$ip" >> $NODE_IP_FILE_PATH

  startTime=$(date +"%s")
  # BATCH_BOOTSTRAP_TIMEOUT is time in minutes which will be used to wait. Default is 15 mins
  if [[ ! -v BATCH_BOOTSTRAP_TIMEOUT ]]; then
    echo " BATCH_BOOTSTRAP_TIMEOUT is not set. Defaulting to 15 minutes."
    waitTime=$(expr 15 \* 60)
  else
    waitTime=$(expr $BATCH_BOOTSTRAP_TIMEOUT \* 60)
  fi



  lines=$(ls ${NODE_IP_FILE_PREFIX}* | wc -l)
  while [ "$AWS_BATCH_JOB_NUM_NODES" -gt "$lines" ]
  do
    currentTime=$(date +"%s")
    duration=$(expr $currentTime - $startTime)
    if [ "$duration" -gt "$waitTime" ] ; then
	    echo "Waited for $duration seconds . Exiting"
	    echo "1" > $AWS_BATCH_EXIT_CODE_FILE
	    kill  $(cat /tmp/supervisord.pid)
            exit 1
    fi
    log "$lines out of $AWS_BATCH_JOB_NUM_NODES nodes joined, waited for $duration.  check again in 30 second"
    sleep 30

    lines=$(ls ${NODE_IP_FILE_PREFIX}* | wc -l)
  done
  # Make the temporary file executable and run it with any given arguments
  log "All nodes successfully joined"

  # remove duplicates if there are any.
  cat $(ls ${NODE_IP_FILE_PREFIX}*) > $HOST_FILE_PATH
  awk '!a[$0]++' $HOST_FILE_PATH > ${HOST_FILE_PATH}-deduped
  cat $HOST_FILE_PATH-deduped
  sudo mkdir -p /job/
  sudo chmod 777 /job
  cp  ${HOST_FILE_PATH}-deduped /job/hostfile
  cat /job/hostfile

  if [[ ! -v BATCH_ENTRY_SCRIPT ]]; then
    echo "BATCH_ENTRY_SCRIPT  is not set. continuing"
  else
    $BATCH_ENTRY_SCRIPT
    if [ $? -eq 0 ]
    then
       log "Writing exit code 0  to $AWS_BATCH_EXIT_CODE_FILE and shutting down supervisord"
       echo "0" > $AWS_BATCH_EXIT_CODE_FILE
    else
       log "Writing exit code 1  to $AWS_BATCH_EXIT_CODE_FILE and shutting down supervisord"
       echo "1" > $AWS_BATCH_EXIT_CODE_FILE
    fi
  fi

  kill  $(cat /tmp/supervisord.pid)
  exit 0

}


# Fetch and run a script
report_to_master () {
  # get own ip and num cpus
  #
  ip=$(/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1)

  if [ -x "$(command -v nvidia-smi)" ] ; then
      NUM_GPUS=$(ls -l /dev/nvidia[0-9] | wc -l)
      availablecores=$NUM_GPUS
  else
      availablecores=$(nproc)
  fi

  log "I am a child node -> $ip:$availablecores, reporting to the master node ->
${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}"
  until echo "$ip" | ssh ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS} "cat > $NODE_IP_FILE_PATH"
  do
    echo "Sleeping 5 seconds and trying again"
  done

  while :
  do
	  echo "Sleeping"
	  sleep 30
  done




  log "done! goodbye"
  exit 0
  }


# Main - dispatch user request to appropriate function
log $NODE_TYPE
install_deps
case $NODE_TYPE in
  main)
    wait_for_nodes "${@}"
    ;;

  child)
    report_to_master "${@}"
    ;;

  *)
    log $NODE_TYPE
    usage "Could not determine node type. Expected (main/child)"
    ;;
esac
