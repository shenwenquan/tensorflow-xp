import tensorflow as tf

cluster = tf.train.ClusterSpec({
    "canshu": [
        "172.17.0.2:12222",  # /job:canshu/task:0
        "172.17.0.3:12222",  # /job:canshu/task:1
    ],
    "gongzuo": [
        "172.17.0.4:12222",  # /job:gongzuo/task:0
        "172.17.0.5:12222",  # /job:gongzuo/task:1
    ]
})

server = tf.train.Server(cluster, job_name="canshu", task_index=1)

server.join()
