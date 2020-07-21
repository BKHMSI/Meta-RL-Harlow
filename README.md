# Meta-RL: Harlow Visual Fixation Task (PyTorch)

In this repository, I reproduce the results of [Prefrontal Cortex as a Meta-Reinforcement Learning System](https://www.nature.com/articles/s41593-018-0147-8)<sup>1</sup> and [Learning to reinforcement learn](https://arxiv.org/abs/1611.05763)<sup>2</sup> on two variants of the Harlow "learning to learn" visual fixation task. The first variant is one-dimensional version of the task that I created as a sanity check for my implementation, it is still able to showcase the same one-shot learning behavior and is a good starting point for those starting to delve into Meta-RL. The other variant is part of DeepMind's [PsychLab](https://deepmind.com/blog/article/open-sourcing-psychlab) that aims to study specifc cognitive skills of artificial agents by recreating experiments from cognitive pyschology in a virtual environment. Here I am using the task that was inspired by Harlow's experiment in [The Formulation of Learning Sets](https://psycnet.apa.org/record/1949-03097-001) in which a monkey was continually presented with new stimuli requiring learning in its fullest sense to attain the food reward. You will find below a description of the task with results, along with a brief overview of Meta-RL and its connection to neuroscience, as well as details covering the structure of the code.

*Note: I have a related repository on the ["Two-Step" task](https://github.com/BKHMSI/Meta-RL-TwoStep-Task) with episodic LSTMs in case you are interested :)

<table align="center">
    <tr>
        <th>Harlow PsychLab</th>
        <th>Harlow Simple</th>
    </tr>
    <tr>
        <td align="center" width="50%"><img alt="PsychLab Demo" src="assets/Harlow_9500.gif"></td>
        <td align="center" width="50%"><img alt="Simple Demo" src="assets/HarlowSimple_6.gif"></td>
    </tr>
</table>