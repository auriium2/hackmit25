import ray

# Replace with your mapping
ray.init(address="ray://213.173.110.199:19780")

print(ray.cluster_resources())
