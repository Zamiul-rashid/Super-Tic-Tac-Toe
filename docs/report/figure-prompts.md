# Methodology figure generation record

## Final asset

- Saved file: [figures/methodology.png](figures/methodology.png).
- Execution mode: built-in `image_gen` tool; no CLI/API fallback.
- User reference: the attached academic methods figure, used for grouped panel layout, feature stacks, muted module colors, labeled arrows, and typography. Its algorithms, labels, and scientific claims were not adopted.
- Final selected output: `exec-cd24852e-26f2-4feb-aec8-dea5da180f6a.png`, copied into the report from the built-in output directory.
- Earlier decorative and generic drafts were rejected and are not included in this report.
- This is an explanatory figure. All quantitative charts are generated separately from evidence with Matplotlib.

## Generation prompt

```text
Use case: scientific-educational
Asset type: main methodology figure in an academic machine-learning paper.
Input images: the attached user reference is a STYLE AND LAYOUT reference only. Match the TOP reference figure's compact two-row multi-panel research-method diagram: thin charcoal borders, pale gray rounded grouped sections, muted pastel blue encoder, salmon decoder, green data blocks, small feature-map stacks and tree glyphs, clean black labeled arrows, bold panel headings. Do NOT copy its molecular graph, GIBMS/SLG labels, or its algorithms. Do not make cover art, a photorealistic scene, or a bare line of generic boxes. Deliver a crisp high-resolution landscape figure with typography large enough for a paper.

Content must describe this Ultimate Tic-Tac-Toe implementation accurately. Exactly TWO large horizontal bordered panels with internal grouped modules, following the supplied TOP reference's density and visual hierarchy.

TOP PANEL heading '(a) Hierarchical policy–value network'. Four pale-gray groups left to right:
1 'State encoding': a schematic nested 9×9 game board with bold dividers every three cells, arrow to small feature-plane stack labeled '8 × 9 × 9 planes'; compact note '289 state features'.
2 'Micro-to-macro encoder': muted-blue tapered encoder block labeled 'Conv + 2 ResBlocks', feature stack '64 × 9 × 9', arrow labeled 'stride 3' to deeper feature stack '128 × 3 × 3' labeled '3 ResBlocks'.
3 'Decoder & skip fusion': salmon tapered upsampling block labeled 'Transposed conv', 'stride 3'; then concatenate circle and fusion block labeled 'Conv', output '64 × 9 × 9'. Show dashed skip-feature arrow from 64 × 9 × 9 encoder stack into concatenation; also input-plane skip into concatenation. Keep these routed above content, without running through text.
4 'Prediction heads': decoded features branch to 'Policy logits (81)' illustrated by a small UNNUMBERED categorical bar glyph and 'Action values (81)' illustrated by small output cells. Separately, route a black arrow from the MACRO 128 × 3 × 3 feature stack below the decoder into 'State value (1)' labeled 'MLP + tanh'. This state value branch MUST originate before the decoder. Bottom small note: 'GroupNorm + ReLU; tanh value outputs'.

BOTTOM PANEL heading '(b) Population learning and paired evaluation'. Four pale-gray groups left to right:
1 'Game generation': small branching tree icon labeled 'Batched PUCT search'; below it compact opponent bank labeled 'Self-play · AlphaBeta · uttt.ai' and 'History · OpenSpiel · heuristics'. Show arrow into search labeled 'Opponent schedule'. Neural network panel above supplies 'policy, value' to search through a routed connector in the panel gutter.
2 'Replay targets': green stacked sample cards labeled '(state, π, z, Q)' and underneath 'Visits · outcome · action values'. Arrow to 'Symmetry augmentation'. Small optional incoming side arrow from 'Native-search bootstrap'.
3 'Optimization': three small vertically stacked loss boxes labeled 'Policy cross-entropy', 'Value MSE', 'Masked Q MSE', joining '+' and 'AdamW'. A dashed red arrow labeled 'Parameter update' returns from AdamW to the network in panel (a), routed along outer margin, avoiding other labels.
4 'Evaluation': checkpoint document icon labeled 'Frozen checkpoint', two paired board-seat glyphs labeled 'Same opening, swapped sides', then a small report sheet labeled 'W / D / L + uncertainty'. Connect optimizer to frozen checkpoint. Small note 'Fixed budgets; not equal time'.

Constraints: all scientific content above is mandatory, do not invent performance numbers, no benchmark curves, no decorative neural glow, no giant X/O artwork, no 3D perspective board. Use subtle feature-stack depth like the reference only. Clearly distinguish solid forward arrows, dashed skip connections, and dashed red parameter-update path. Avoid tiny crowded text. Spelling and tensor dimensions must be exact. Make this look like the reference's academic methods figure.
```

## Targeted correction prompt

```text
Edit this academic methodology diagram. Preserve its exact two-panel composition, typography, colors, feature stacks, labels, and all existing modules. Make only these scientific wiring corrections: (1) Route the dashed input-planes skip arrow in panel (a) into the C concatenation circle, not into the Conv block. Both dashed skip arrows must enter C. (2) Add a clear solid arrow from the Symmetry augmentation block in panel (b) to the Optimization group, then branch it into all three loss boxes; this arrow must not cross label text. (3) The dashed red Parameter update arrow from AdamW must terminate at the blue Conv + 2 ResBlocks encoder block in panel (a), showing the model parameters are updated. Route that red path through empty margins; remove the current red endpoint at the Replay targets group. Keep every tensor dimension and every other part of the figure unchanged.
```

## Scientific interpretation

Panel (a) summarizes the implemented architecture. Both skip routes supply the fusion stage; the precise tensor concatenation is `[micro_features, upsampled_macro_features, input_planes]`, performed before the fusion convolution. Table 2 of the manuscript is the authoritative layer-by-layer specification. The state-value branch originates at the macro features, while policy and Q heads use decoded features. Panel (b) summarizes target generation and training; the checkpoint route represents saving learned parameters, not a differentiable loss operation. Bars and boards are schematic glyphs and do not encode measured outcomes.

