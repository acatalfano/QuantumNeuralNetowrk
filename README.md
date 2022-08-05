# View In Jupyter Notebook Interface

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
