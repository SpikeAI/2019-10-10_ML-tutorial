
# A tutorial introducing Deep Learning for vision science
## Chloé Pasturel, Laurent Perinet (INT)

In this practical workshop, we propose to present the new challenges brought by deep learning and more generally by machine learning. The objective is to show in the form of simple practical exercises how these new tools allow 1) to categorize images 2) to learn such a model 3) to generate new images from an existing database.

# Atelier méthodologique « Utiliser apprentissage profond en vision »

## Chloé Pasturel, Laurent Perinet (INT)

Nous proposons dans un court tutoriel de présenter les nouveaux enjeux apportés par l'apprentissage profond et plus généralement par l'apprentissage machine. L'objectif est de montrer sous forme de simples exercises pratiques comment ces nouveaux outils permettent 1) de généraliser procédures d'analyse de données, 2) de tester de façon simple des modèles de réponses visuelles.

# some notes

## recent papers


* Engineering a Less Artificial Intelligence https://www.nature.com/articles/d41586-019-02212-4:
Volume 103, Issue 6, 25 September 2019, Pages 967-979  Perspective Fabian H.Sinz XaqPitkow JacobReimer MatthiasBethge Andreas S.Tolias https://doi.org/10.1016/j.neuron.2019.08.034

* Call for papers https://www.journals.elsevier.com/vision-research/call-for-papers/what-do-deep-neural-networks-tell-us-about-biological-vision Recent years have seen a huge increase in the application of deep learning techniques and ‘biologically inspired’ deep neural networks (DNNs) to a broad range of issues in biological vision. Indeed, DNNs have been described by some as a new framework for vision research, allowing an opportunity to ‘reverse engineer’ the biological system. These claims are, in part, based on work showing human-level performance by DNNs in tasks such as image classification and are supported by advances in the development of methods for comparing representational structures computed by DNNs with biological vision systems. But the suitability of such networks as a theoretical framework for understanding biological vision is unclear. There remain many important questions: How should theoretically relevant and irrelevant properties of DNN architectures and processing parameters be distinguished? How can network performance be rigorously compared with corresponding biological data? What is the range of relevant performance data for evaluating network outputs? And to what extent can network activity be used to formulate empirically testable models of biological vision? This special issue invites novel contributions on these and related topics. We welcome original articles that consider the application of DNNs to understanding any aspect of biological vision. We particularly welcome contributions that provide a critical evaluation of DNNs as models of human vision

* Headline review *Opportunities and obstacles for deep learning in biology and medicine* https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0387


## background
Artificial neural networks were originally conceived as a model for computation in the brain [7]. Although deep neural networks have evolved to become a workhorse across many fields, there is still a strong connection between deep networks and the study of the brain. The rich parallel history of artificial neural networks in computer science and neuroscience is reviewed in [346–348].
CNNs were originally conceived as faithful models of visual information processing in the primate visual system, and are still considered so [349]. The activations of hidden units in consecutive layers of deep convolutional networks have been found to parallel the activity of neurons in consecutive brain regions involved in processing visual scenes. Such models of neural computation are called ‘encoding’ models, as they predict how the nervous system might encode sensory information in the world.
Even when they are not directly modelling biological neurons, deep networks have been a useful computational tool in neuroscience. They have been developed as statistical time-series models of neural activity in the brain. And in contrast to the encoding models described earlier, these models are used for decoding neural activity, for instance, in brain–machine interfaces [350]. They have been crucial to the field of connectomics, which is concerned with mapping the connectivity of biological neural networks in the brain. In connectomics, deep networks are used to segment the shapes of individual neurons and to infer their connectivity from 3D electron microscopic images [351], and they have also been used to infer causal connectivity from optical measurement and perturbation of neural activity [352].
It is an exciting time for neuroscience. Recent rapid progress in deep networks continues to inspire new machine learning-based models of brain computation [346]. And neuroscience continues to inspire new models of artificial intelligence [348].




Machine learning methods enable researchers to discover statistical patterns in large datasets to solve a wide variety of tasks, including in neuroscience. Recent advances have led to an explosion in the scope and complexity of problems to which machine learning can be applied, with an accuracy rivaling or surpassing that of humans in some domains.
This virtual conference will illuminate the many ways machine learning and neuroscience intersect in the context of data analysis and modeling brain function, and how neuroscience can benefit from the machine learning revolution.

Topics include:
* Basic machine learning concepts and resources.
* Machine learning methods to automate analyses of large neuroscience datasets.
* Using deep network learning to gain insight into how the brain learns.
* Combining machine learning concepts with neuroscience theory to predict nervous system function and uncover general principles.
The conference will end with speakers sharing their views on promising future directions for both machine learning and neuroscience.



* intro by bengio = https://interstices.info/la-revolution-de-lapprentissage-profond/

* Jürgen Schmidhuber: "Deep Learning in Neural Networks: An Overview," Neural Networks, volume 61, pp. 85-117, January 2015.


* The term machine learning was coined by Arthur Samuel in 1959 to describe the subfield of computer science that involves the “programming of a digital computer to behave in a way which, if done by human beings or animals, would be described as involving the process of learning” (Samuel, 1959). 

* However, the current models are designed with engineering goals and not to model brain computations.  it's a set-back from the computational neuroscience approach / highly successful / but adversarial examples

* "The deep mystery of vision: How to integrate generative and discriminative models"



* Whittington19 
```
@article{Whittington19,
  title = {Theories of {{Error Back}}-{{Propagation}} in the {{Brain}}},
  volume = {23},
  issn = {13646613},
  doi = {10.1016/j.tics.2018.12.005},
  language = {en},
  number = {3},
  journal = {Trends in Cognitive Sciences},
  author = {Whittington, James C.R. and Bogacz, Rafal},
  month = mar,
  year = {2019},
  pages = {235-250},
  file = {/Users/laurentperrinet/Zotero/storage/EQCWD575/Whittington and Bogacz - 2019 - Theories of Error Back-Propagation in the Brain.pdf},
  note = {00001}
}
```

* https://neuronline.sfn.org/Articles/Scientific-Research/2019/Machine-Learning-in-Neuroscience-Fundamentals-and-Possibilities?utm_campaign=Machine%20Learning%20Virtual%20Conference


## related events

* on CVnet : "As part of the Summer School on Imaging with Medical Applications SSIMA 2019, 16-20 September 2019 in Bucharest, Romania, a full-day masterclass will be given on 16 September 2019 by prof. Bart ter Haar Romeny, Eindhoven University of Technology, the Netherlands. See for all details and registration: https://ssima.eu/masterclass-deep-learning/.
 
The Masterclass on Deep Learning will not only discuss the layers of the convolutional networks and the application of modern Deep Learning tools, but will go further. We will discuss the intrinsic mechanisms of the black box: the essential notion of representation learning, and the neuro-mathematics of self-organization and contextual processing. We will take a deep tour into modern vision and brain research, both in the retina as visual cortex, and realize the strong similarities between the visual pathway and Deep Learning, and how much can be learned from one by studying the other. We demonstrate surgical software tools to dissect the network layers and look inside what they compute, and explain modern visualization tools, like t-SNE.
All essential mathematics is explained in an intuitive way, with many visual and hands-on examples.
The masterclass can be followed by all interested in deep learning, with an emphasis on trying to understand what happens inside. All code of all examples is given to the participants.
 
The Summer School is MICCAI endorsed, supported by EuSoMII and has world-class speakers on Deep Learning in Medical Imaging, Big Data in Healthcare, Medical Sensors, Robots and AI-based Data Analysis.
Website: https://ssima.eu. "

Organizers

* Kristin Branson is a group leader and the head of computation and theory at the Howard Hughes Medical Institute's Janelia Research Campus. Her lab develops new machine vision and learning technologies to extract scientific understanding from large image data sets. Using these systems, Janelia Research Campus aims to gain insight into behavior and how it is generated by the nervous system. She earned her BA in computer science from Harvard University and her PhD in computer science from the University of California, San Diego, and completed postdoctoral training at the California Institute of Technology.

* Edda (Floh) Thiels is an adjunct associate professor of neurobiology at the University of Pittsburgh School of Medicine and a program director in the Directorate for Biological Sciences at the National Science Foundation. Thiels’ main research interests lie in how animals acquire information from the environment and use that information to guide their behavior. She received her undergraduate degree in psychology from the University of Toronto and her PhD in psychology from Indiana University.


## Réunion du GdR ISIS
* Titre : Théorie du deep learning
* Dates : 2019-10-17
* Lieu : Cnam Paris, amphithéâtre Paul Painlevé - 292, rue Saint Martin 75003 Paris.
Annonce :
Les réseaux de neurones profonds ont marqué l'entrée dans une nouvelle ère de l'intelligence artificielle, ponctuée par des succès opérationnels dans des domaines variés de la science des données comme la classification d'images, la reconnaissance vocale, ou le traitement de la langue naturelle. 
En dépit de ces succès importants, les garanties théoriques associées à ces modèles décisionnels restent aujourd'hui toujours fragiles. L'objectif de cette journée est de faire un état des lieux sur la compréhension du fonctionnement des réseaux de neurones profonds, à travers un appel à contributions centré autour les thèmes (non exhaustifs) suivants :

** Expressivité des modèles
** Robustesse décisionnelle (incertitude, stabilité, attaques adversaires)
** Optimisation et problèmes non convexes
** Théorie de la généralisation
** Lien entre modèles physiques et architectures de réseaux de neurones

Les outils utilisés pour aborder ces thématiques pourront venir de l'apprentissage statistique, mais des méthodes venant de disciplines connexes (décomposition tensorielles, analyse harmonique, méthodes géométriques / algébriques, physique statistique) sont fortement encouragées.
Orateurs inivtés :
* Rémi Gribonval, LIP ENS Lyon,
* Edouard Oyallon, LIP6 Paris, CNRS

Appel à contributions :
Les personnes souhaitant présenter leurs travaux à cette journée sont invitées à envoyer, par e-mail, leur proposition (titre et résumé de 1 page maximum) aux organisateurs avant le 26 septembre 2019.
Organisateurs :
* Caroline Chaux-Moulin (valentin.emiya@lis-lab.fr), Université Aix-Marseille, I2M
* Valentin Emiya (caroline.chaux@univ-amu.fr), Université Aix-Marseille, LIS
* François Malgouyres (Francois.Malgouyres@math.univ-toulouse.fr), Institut de Mathématiques de Toulouse (IMT, CNRS UMR 5219)
* Nicolas Thome (nicolas.thome@cnam.fr), Cnam Paris
* Konstantin Usevich (konstantin.usevich@univ-lorraine.fr), Université de Lorraine, CRAN, Nancy

Lien : http://gdr-isis.fr/index.php?page=reunion&idreunion=405
