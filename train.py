import os
import json
import pickle
import argparse
import numpy as np
import shap
import torch
import gym
import pandas as pd

from sklearn.cluster import KMeans
from models.nets import Expert
from models.gail import GAIL




import lime
import lime.lime_tabular


def main(env_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in ["CartPole-v1", "Pendulum-v0", "BipedalWalker-v3"]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(
        state_dim, action_dim, discrete, **expert_config
    ).to(device)
    expert.pi.load_state_dict(
        torch.load(
            os.path.join(expert_ckpt_path, "policy.ckpt"), map_location=device
        )
    )

    model = GAIL(state_dim, action_dim, discrete, config).to(device)

    a = pd.read_csv('a.csv')
    s = pd.read_csv('s.csv')
    a = a.drop(columns=['Unnamed: 0'])
    s = s.drop(columns=['Unnamed: 0'])
    r = pd.read_csv('r.csv')
    rev = int(r.iloc[0,1])



    results = model.train(env, expert, exp_obs=s, exp_acts=a, r=rev)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )
    if hasattr(model, "d"):
        torch.save(
            model.d.state_dict(), os.path.join(ckpt_path, "discriminator.ckpt")
        )

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    args = parser.parse_args()

    mod, res = main(**vars(args))


    data = np.array(res[4])
    print(len(data))

    probs = np.array(mod.pi(res[4]).probs.detach())

    kmeans = KMeans(n_clusters=8, random_state=0).fit(data)

    def prob(data):
        data = torch.tensor(data, dtype=torch.float32)
        return np.array(mod.pi(data).probs.detach())


    explainer = lime.lime_tabular.LimeTabularExplainer(data,
                                                       mode='classification',
                                                       class_names= ['left', 'right'],
                                                       feature_names=['position of cart', 'velocity of cart', 'angle of pole', 'rotation rate of pole'],
                                                       training_labels=np.array(mod.pi(res[4]).probs.detach()))


    # asking for explanation for LIME model
    i = 1
    exp = explainer.explain_instance(kmeans.cluster_centers_[0,:], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[0,:],prob(kmeans.cluster_centers_[0,:])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[1,:], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[1,:],prob(kmeans.cluster_centers_[1,:])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[2,:], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[2,:],prob(kmeans.cluster_centers_[2,:])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[3,:], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[3,:],prob(kmeans.cluster_centers_[3,:])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[4,:], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[4,:],prob(kmeans.cluster_centers_[4,:])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[5, :], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[5, :], prob(kmeans.cluster_centers_[5, :])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[6, :], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[6, :], prob(kmeans.cluster_centers_[6, :])))
    exp.as_pyplot_figure().show()

    exp = explainer.explain_instance(kmeans.cluster_centers_[7, :], prob, num_features=4)
    print("feats: {}, probs: {}".format(kmeans.cluster_centers_[7, :], prob(kmeans.cluster_centers_[7, :])))
    exp.as_pyplot_figure().show()



    def shap_prob(data):
        data = torch.tensor(data, dtype=torch.float32)
        return np.array(mod.pi(data).probs[:,1].detach())

    explainer = shap.KernelExplainer(shap_prob, data=data)
    samp = shap.sample(data, 100)

    shap_values = explainer.shap_values(X=samp)

    shap.summary_plot(shap_values=shap_values, features=samp,
                      feature_names=['position of cart',
                                     'velocity of cart',
                                     'angle of pole',
                                     'rotation rate of pole'],
                      class_names=['left','right'])





    print(res)
    print(mod)



