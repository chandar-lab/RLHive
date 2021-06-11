import argparse
import yaml

def make_changes(args, input_config):
	if args.algo == "decentralized":
		input_config["self_play"] = False
	elif args.algo == "selfplay":
		input_config["self_play"] = True
	else:
		assert True, "Algorithm should be `decentralized` or `selfplay`. Given {}".format(args.algo)

	input_config["run_name"] = args.exp_name
	for i, logger in enumerate(input_config["loggers"]):
		if logger["name"] == "WandbLogger":
			input_config["loggers"][i]["kwargs"]["run_name"] = args.exp_name
			input_config["loggers"][i]["kwargs"]["save_dir"] = args.output_directory
	input_config["save_dir"] = args.output_directory
	input_config["stack_size"] = args.stack_size
	input_config["environment"]["kwargs"]["env_name"] = args.env_name
	input_config["environment"]["kwargs"]["seed"] = args.seed
	AGENT_SEED_GAP = 13 #initialize different agents with seeds different by this value
	for i, agent in enumerate(input_config["agents"]):
		input_config["agents"][i]["kwargs"]["replay_buffer"]["kwargs"]["stack_size"] = args.stack_size
		input_config["agents"][i]["kwargs"]["replay_buffer"]["kwargs"]["seed"] = args.seed + i*AGENT_SEED_GAP
		input_config["agents"][i]["kwargs"]["seed"] = args.seed + i*AGENT_SEED_GAP
		#learning rates - "lr" is the argument for many torch.optim optimizers
		#scale lr of agents in a decreasing geometric fashion
		input_config["agents"][i]["kwargs"]["optimizer_fn"]["kwargs"]["lr"] = args.base_lr / args.lr_scale_factor**i
	return input_config

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-od", "--output-directory")
	parser.add_argument("-ic", "--input-config")
	parser.add_argument("-oc", "--output-config")
	parser.add_argument("--env_name", help="Name of the Gym Env used in `gym.make`")
	parser.add_argument("--exp_name", help="Experiment name")
	parser.add_argument("--algo", help="Multi-agent algorithm used: `decentralized` or `selfplay`")
	parser.add_argument("--stack_size", type=int, help="Number of frames to stack")
	parser.add_argument("--base_lr", type=float, help="Base learning rate of the first agent")
	parser.add_argument("--lr_scale_factor", type=float, help="Scale down lrs by this factor >= 1.0")
	parser.add_argument("--seed", type=int, help="Random Seed to use in the experiment")

	args = parser.parse_args()

	with open(args.input_config) as f:
		input_config = yaml.safe_load(f)

	input_config = make_changes(args, input_config)

	with open(args.output_config, 'w+') as f:
		yaml.safe_dump(input_config, stream=f)