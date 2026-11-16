# Security oracle: scenario cards and calibration controls

`cards.yaml` (`scenario-cards-v1`, query map `qmap-v1`) holds one card for each
of the 35 LLMSecEval families of the sample: the weakness, the CodeQL queries
that count for the targeted measure, the calibration state and the controls.

`controls/` holds the calibration programs, each named by the cards with its
SHA-256:

- `V__<family>__*.py`: a common way to write the weakness of the scenario;
- `S2__<family>__*.py`: the recommended safe idiom;
- `B__<family>__*.py`: a known blind spot of the analyser, expected not to be reported;
- `M__*.py`: probes (a program that does nothing, a stub, and similar).

**Warning.** The `V__` and `B__` files are vulnerable on purpose. They exist
only to check that the analyser reports what it should. Do not run them and do
not reuse them.

The secure implementations shipped with LLMSecEval are used as further
controls of specificity. They are not copied here: the cards name them by path
inside `Dataset/Secure Code Samples` of the pinned dataset revision and by
SHA-256.

Linting and formatting tools skip this folder: a control must stay byte for
byte the program whose hash is recorded.
