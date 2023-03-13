# Odor detection project

![desert_locust](docs/desert_locust.jpeg)

## Summary
Classification of odor's signals that where recorded using the desert locust antennae.

This project is a collaboration with Neta Shvil a PhD student in Neuroscience in Tel Aviv university. During her PhD she studied how antenas on locuses record and identify smells. Here we would like to continue her work with the intent to improve the ordor classification. Her experiment consist of 1 or 2 antenas cut of (on both sides) from an locus placed in a conductive gel and on which 2 electrodes are mounted on each extremity. This is recorded with a potentiometer and stored as a digital signal. A smell is then send in the direction of the antenna producing a stimuli by the antenna and recorded at 100Hz. Hence, the data consist of a spectral recording of differents smells. In total 8 smells were tested in this manner.

For more information the published article is liked at: [The Locust antenna as an odor discriminator](https://www.sciencedirect.com/science/article/abs/pii/S0956566322009599)

## Data dictionary
### Labels
- Benz: Benzaldehyde
- Hex: 1-Hexanol
- Ethyl: Ethyl-butyrate
- Rose: Rosemary
- Lem: Lemon
- Ger: Geraniol
- Cit: $\beta$-citronellol
- Van: Vanilla

### Features
150 data points per smell and antennae spans over 1.5 second of recording.