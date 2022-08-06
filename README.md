# Quantum Neural Network in Q#

## View In Jupyter Notebook Interface

Install anaconda then create and activate an environment:

```bash
conda create -n <env name> -c microsoft qsharp notebook
conda activate <env name>
```

Then run these install commands (tested in an anaconda environment). Confirm when prompted

```bash
conda install numpy pytorch
conda install -c pytorch torchvision
```

Now run `jupyter notebook ./QuantumNeuralNet.ipynb` from the root of this repository.

Once Jupyter Notebook is loaded in a browser, you can click into `Kernel > Restart & Run All`
and see the bulk of the work that constitutes the neural net.

I did not finish the whole implementation. The entirety of the quantum logic is outlined in the jupyter notebook,
but a fully functional feature is lacking, mostly because I did not become acquainted with pytorch quickly enough.

My preliminary (nearly finished) classical implementations are included, but are far from runnable.

## References

[Elementary gates for quantum computation](https://arxiv.org/pdf/quant-ph/9503016.pdf)
was used for designing the generic parameterized gate

[Circuit-centric quantum classifiers](https://arxiv.org/pdf/1804.00633.pdf)
presented a design for a binary quantum classifier, which was adapted to be
the core of this non-binary neural net design

[Gradients of parameterized quantum gates
    using the parameter-shift rule and gate decomposition
](https://arxiv.org/pdf/1905.13311.pdf)
was leveraged for defining how the gradient of the quantum circuit could be approximated,
which was instrumental for applying backpropagation

[Hybrid quantum-classical Neural Networks with PyTorch and Qiskit
](https://qiskit.org/textbook/ch-machine-learning/machine-learning-qiskit-pytorch.html)
provides a concrete implementation, from which the classical computing part of this algorithm is adapted
