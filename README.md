# Learning to Weight Similarity Measures with Siamese Networks: A Case Study on Optimum-Path Forest

*This repository holds all the necessary code to run the very-same experiments described in the chapter "Learning to Weight Similarity Measures with Siamese Networks: A Case Study on Optimum-Path Forest".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `data`: Folder containing the OPF file format datasets;
 * `outputs`: Folder for saving the output files, such as `.npy`, `.pkl` and `.txt`;
 * `utils`
   * `loader.py`: Loads OPF file format datasets;
   * `similarity.py`: Calculates the similarity between pairs of samples.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

Please [download](https://www.recogna.tech/files/opf_siamese/data.tar.gz) the datasets in the OPF file format and put then on the `data/` folder.

---

## Usage

### Learn a Similarity Function

The first step is to learn a similarity function based on the training data and create the dataset's squared similarity matrix. To accomplish such a step, one needs to use the following script:

```Python
python learn_similarities.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Perform the Classification

After learning the similarity, one needs to classify the data using an OPF-based classifier or Scikit-Learn classifiers. Please, use the following scripts to accomplish such a procedure:

```Python
python classify_with_opf.py -h
```

or

```Python
python classify_without_opf.py -h
```

### Process Classification Reports

After conducting the classification task, one needs to process its report into readable outputs. Please, use the following script to accomplish such a procedure:

```Python
python process_report.py -h
```

*Note that this script converts the .pkl reports into readable .txt outputs.*

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this chapter. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
