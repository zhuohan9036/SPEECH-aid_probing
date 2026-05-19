from parameters import parse

args = parse()


print(f"Dataset name: {args.dataset_name}")
print(f"Dataset path: {args.dataset_path}")