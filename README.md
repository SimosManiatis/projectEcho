## Project Overview

Decision-making in early-stage design often lacks robust methods for evaluating circularity, resulting in outcomes that may not fully realize their potential for efficiency. This research presents the development of a computational tool or “Intelligent Design Assistant” that employs Graph Neural Networks (GNNs) to deliver real-time assessments of life-cycle performance and material usage for modular designs. By utilizing user-defined, simplified early-stage representations, the tool provides actionable insights into both design and environmental performance. A central point of this approach is the adoption of a graph-based framework where each building module is represented as a node, and its interactions with neighboring modules are captured through connecting edges. This framework not only reflects the intrinsic properties of each module, but it also dynamically evaluates how a module’s characteristics evolve based on its spatial and functional relationships. Although the study focuses on laminated veneer lumber (LVL)—selected for its extensive environmental data—the scalable machine learning model is designed to be applicable to a wide range of construction methods and materials. Through experimental validation, the integration of GNNs has been shown to enhance early design decision-making by providing real-time feedback. The model achieves an accuracy of approximately 85% -90% under conditions similar to the training data. This capability enables designers, clients, and other stakeholders to engage in informed discussions about design modifications and circularity measures well before detailed construction planning begins, thereby promoting more sustainable and circular design practices across the industry.

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
- `gh/`: Grasshopper files which contain the the developed components of the plugin.

This repository provides access to the three main components of the project:

1) **The Assembly Sequence Algorithm**

The core structure of this computational framework is integrated into all parts of the project.

**Required Files:**
  • [assembly_sequence_and_cutting_stock.gh](https://github.com/SimosManiatis/multi-robot-assembly/blob/main/gh/assembly_sequence/assembly_sequence_and_cutting_stock.gh)

2) **The Structure Design & Optimisation Algorithm**

**Required Files:**

  • [241105_Combined model.gh](https://github.com/SimosManiatis/multi-robot-assembly/blob/main/gh/structural_optimisation/241105_Combined%20model.gh)
  • [241105_Combined model.3dm](https://drive.google.com/file/d/1Ds01K0zyoJiDR4t8Iob-g5H5hhUbotOR/view?usp=sharing)  
  • [on_site_assembly.xlsx](https://github.com/SimosManiatis/multi-robot-assembly/blob/main/gh/structural_optimisation/on_site_assembly.xlsx)

3) **The Multi-Robot Assembly Setup Algorithm**

**Required Files:**

  • [RoboticAssemblyPrototype.3dm](https://drive.google.com/file/d/1RJFRhfzesO6kzf9160D5oSREZPiYviEM/view?usp=drive_link)  
  • [RoboticAssemblyPrototype.rdk](https://drive.google.com/file/d/1FT9tKub4zBwlEuxIuXIC0ZLAkGPYGocG/view?usp=drive_link)  
  • [RoboticAssemblyPrototype.gh](https://github.com/SimosManiatis/multi-robot-assembly/blob/main/gh/robotic_assembly/RoboticAssemblyPrototype.gh)

**Optional External Interface:**
  • [RobotManager.py](https://github.com/SimosManiatis/multi-robot-assembly/blob/main/src/robotic_assembly/RobotManager.py)

---

### Prerequisites

- **Rhinoceros 8**  
  Download from [https://www.rhino3d.com/](https://www.rhino3d.com/)

- **RoboDk**  
  Download from [https://robodk.com/download](https://robodk.com/download)

- **Rhinoceros & RoboDk Files**  
  Download from [Google Drive](https://drive.google.com/drive/folders/1hj0ywdX9TM16v8JOCXHmmRV7WWfTj7M6?usp=drive_link)

- **GH_Linear_Cutting**  
  Download .gha file from [https://github.com/AlexanderMorozovDesign/GH_Linear_Cutting](https://github.com/AlexanderMorozovDesign/GH_Linear_Cutting)

- **Karamba**  
  Download from [https://karamba3d.com/](https://karamba3d.com/)

- **Hops**  
  Download from [https://www.food4rhino.com/en/app/hops](https://www.food4rhino.com/en/app/hops)

- **Telepathy**  
  Download from [https://www.food4rhino.com/en/app/telepathy](https://www.food4rhino.com/en/app/telepathy)

- **Human**  
  Download from [https://www.food4rhino.com/en/app/human](https://www.food4rhino.com/en/app/human)

- **Fox**  
  Download from [https://www.food4rhino.com/en/app/fox](https://www.food4rhino.com/en/app/fox)

- **Pufferfish**  
  Download from [https://www.food4rhino.com/en/app/pufferfish](https://www.food4rhino.com/en/app/pufferfish)

---

### Installation

1. Clone the repository:
   ```bash
   gh repo clone SimosManiatis/multi-robot-assembly

2. Install Pre-requisites
**Rhinoceros 8,RoboDk & Karamba need licenses ro run.**

### Contact

For questions or support, please contact:
**Email** : smaniatis@tudelft.nl , P.Feijen@student.tudelft.nl, s.a.bentvelsen@student.tudelft.nl, A.Mavrotas@student.tudelft.nl

### Citation 
[GH_Linear_Cutting GitHub Repository](https://github.com/AlexanderMorozovDesign/GH_Linear_Cutting)
