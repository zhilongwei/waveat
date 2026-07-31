# WaveAt: Wave Attenuation by Vegetation

WaveAt is a Python package for predicting the attenuation of regular and
irregular waves as they propagate through a submerged vegetation canopy.

## Installation

Clone the repository and create the locked development environment with
[`uv`](https://docs.astral.sh/uv/):

```console
git clone https://github.com/zhilongwei/waveat.git
cd waveat
uv sync
```

Run Python inside that environment with `uv run python`. To install the package
from an existing checkout without `uv`, use a Python 3.14 environment:

```console
python -m pip install .
```

## Reference implementation

The original implementation of the wave attenuation model is available in the DTU GitLab repository [`waveattenuationbyseaweed`](https://gitlab.gbar.dtu.dk/floatingseaweedfarms/waveattenuationmodels/waveattenuationbyseaweed).
WaveAt extends the model in a unified Python package with a consistent API.

## Related publications

1. Shao, Y., Weiss, M., & Wei, Z. (2024). [Wave Attenuation by Cultivated
   Seaweeds: a Linearized Analytical Solution](http://www.iwwwfb.org/Abstracts/iwwwfb39/IWWWFB39_39.pdf).
   In *Proceedings of the 39th International Workshop on Water Waves and
   Floating Bodies*.

2. Wei, Z., Weiss, M., Kristiansen, T., Kristiansen, D., & Shao, Y. (2025).
   [Wave attenuation by cultivated seaweeds: A linearized analytical
   model](https://doi.org/10.1016/j.coastaleng.2024.104642). *Coastal
   Engineering, 195*, 104642.

3. Wei, Z., & Shao, Y. (2026). [Wave Attenuation by Suspended Flexible
   Vegetation](http://www.iwwwfb.org/Abstracts/iwwwfb41/IWWWFB41_52.pdf).
   In *Proceedings of the 41st International Workshop on Water Waves and
   Floating Bodies*, article 52.

4. Wei, Z., Boyer, R., & Shao, Y. (2026). [Analytical Solutions of Two-Way
   Coupled Wave-Flexible Seagrass
   Interaction](https://orbit.dtu.dk/en/publications/analytical-solutions-of-two-way-coupled-wave-flexible-seagrass-in/).
   In *Proceedings of the 10th International Conference on Hydroelasticity in
   Marine Technology*.

## References

1. Dalrymple, R. A., Kirby, J. T., & Hwang, P. A. (1984). [Wave diffraction due
   to areas of energy
   dissipation](https://doi.org/10.1061/(ASCE)0733-950X(1984)110:1(67)).
   *Journal of Waterway, Port, Coastal, and Ocean Engineering, 110*(1), 67–79.

2. Kobayashi, N., Raichle, A. W., & Asano, T. (1993). [Wave attenuation by
   vegetation](https://doi.org/10.1061/(ASCE)0733-950X(1993)119:1(30)).
   *Journal of Waterway, Port, Coastal, and Ocean Engineering, 119*(1), 30–48.

3. Jacobsen, N. G. (2016). [Wave-averaged properties in a submerged canopy:
   Energy density, energy flux, radiation stresses and Stokes
   drift](https://doi.org/10.1016/j.coastaleng.2016.07.009). *Coastal
   Engineering, 117*, 57–69.

4. Zhu, L. (2020). [Wave Attenuation Capacity of Suspended Aquaculture
   Structures with Sugar Kelp and
   Mussels](https://digitalcommons.library.umaine.edu/etd/3222/). PhD thesis,
   University of Maine.

## License

WaveAt is distributed under the [MIT License](LICENSE).
