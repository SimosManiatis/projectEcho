## Project Overview

Early-stage design often lacks solid tools for assessing circularity, risking inefficient outcomes. We introduce an “Intelligent Design Assistant” powered by Graph Neural Networks (GNNs) that delivers real-time life-cycle and material-use feedback on modular designs. Using simplified representations defined by the user, the tool models each module as a graph node and captures its interactions via edges, dynamically tracking how its properties change in context. Although our experiments focus on laminated veneer lumber (LVL) for its rich environmental data, the scalable model works across materials and construction methods. Validated against similar data, it achieves 85–90% accuracy, enabling designers and stakeholders to explore sustainable, circular decisions long before detailed planning.

**Make sure you check the requirements**
[Requirements](https://github.com/SimosManiatis/projectEcho/blob/main/requirements.txt)

**Read the full documentation of the project:**
[projectEcho_MSc Thesis.pdf](https://github.com/SimosManiatis/projectEcho/blob/main/projectEcho_MSc%20Thesis.pdf)

(Open the .pdf file externally since GitHub may not display all the images properly)

**Demo Files**
[Examples](https://github.com/SimosManiatis/projectEcho/tree/main/demoFiles)

## Repository Structure

The repository is organized as follows:
- `src/`: Contains the source code for the Graph Neural Network
- `demoFiles/`: Example files for the showcase of the Grasshopper Components
- `dataset/`: The data the neural network was trained on (ID + OOD).

---

This repository provides access to:

1) **Project Documentation**
Here you can access and review all of the research and development of the tool:
2) **The Graph Neural Network**
In the  `src/` you can find all of the source code of the developed GNN, which provides the material usage prediction.
3) **The EPD Database**
In this excel file, the data from 3 different European manufacturers is documented, which is later pulled from the GH plugin to create the necessary feedback.
4) **The Grasshopper Components [Not Built]**
In the main GH file, all the developed components are situated in the form of clusters.

---

### Prerequisites

- **Rhinoceros 8**  
  Download from [https://www.rhino3d.com/](https://www.rhino3d.com/)

---

### Installation

1. Clone the repository:
   ```bash
   gh repo clone SimosManiatis/projectEcho

2. Install Pre-requisites
**Rhinoceros 8,RoboDk & Karamba need licenses ro run.**

### Contact

For questions or support, please contact:
**Email** : smaniatis@tudelft.nl 

