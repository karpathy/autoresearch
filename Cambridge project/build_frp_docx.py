#!/usr/bin/env python3
"""Build Round 14 Full Research Proposal Word document."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

OUT = "/Users/mehmetaltay/Documents/GitHub/autoresearch/Cambridge project/FRP_R14_MAltay.docx"

doc = Document()
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run(
    "Adaptive Computerized Dynamic Assessment: Synthesizing Eye-Tracking and Reader Profiles "
    "for Inclusive English Reading Comprehension"
)
run.bold = True
run.font.size = Pt(14)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run(
    "Full Research Proposal — Cambridge English Funded Research Programme (Round 14)"
).italic = True

sections = [
    (
        "Alignment with the Round 14 call",
        (
            "This proposal responds to the Round 14 theme of exploring emerging constructs in language assessment "
            "by operationalising “construct-relevant process evidence” alongside product scores. Rather than "
            "treating comprehension as a single latent trait inferred only from item responses, the project "
            "models reading as a dynamic interaction among text features, learner profiles, and moment-to-moment "
            "processing behaviours that can be indexed through eye-movement metrics and responsiveness to "
            "calibrated mediation. The design therefore targets construct representation and explanatory "
            "fairness in digital reading assessment: it connects observable processing signatures to adaptive "
            "scaffolding aligned with the Zone of Proximal Development (ZPD), supporting more inclusive "
            "inferences for diverse EFL readers."
        ),
    ),
    (
        "Background and context of the study",
        (
            "Over a decade ago, McNamara (2014) argued that language assessment was facing a moment of crisis, "
            "driven by technological advances and shifting communicative realities. Today, that call for "
            "re-evaluation is more urgent. As educational landscapes evolve, assessment must move from "
            "static snapshots toward innovative, inclusive practices that reflect the complexity of learning. "
            "A productive framework is Vygotskyan Dynamic Assessment (DA), which reconceptualises testing and "
            "teaching as a single, dialectically integrated activity (Lantolf & Poehner, 2010; Poehner & Wang, 2021). "
            "Rooted in sociocultural theory and the ZPD, DA integrates assessment and instruction so that the goal "
            "is not only to measure independent performance but to evaluate emerging abilities through cooperative "
            "mediation (Poehner & Lantolf, 2005). In EFL reading comprehension, DA identifies where a reader "
            "struggles and provides immediate, calibrated support.\n\n"
            "Historically, DA divides into Interactionist and Interventionist traditions (Lantolf & Poehner, 2004). "
            "Interactionist DA is highly responsive but labour-intensive; Interventionist DA scales through "
            "standardised prompts. Computerised formats (C-DA) extend Interventionist models to larger cohorts "
            "(Poehner et al., 2015). Recent work on hybrid computerised DA further shows how principled mediation "
            "design can combine interactionist insights with scalable delivery in L2 reading contexts (Jin & Liu, 2024). "
            "However, conventional C-DA still largely delivers fixed mediation sequences and therefore under-uses "
            "information about how individual learners process text in real time.\n\n"
            "Eye-tracking provides millisecond-level evidence of allocation of attention during reading and has "
            "become a mainstream complement to outcome measures in second language research (Godfroid, Winke, & Conklin, 2020). "
            "Large-scale corpora such as MECO L2 also demonstrate substantial heterogeneity in the determinants of L2 "
            "reading fluency versus comprehension accuracy across diverse L1 backgrounds, underscoring the need for "
            "assessment models that recognise multiple reader profiles rather than a single developmental pathway "
            "(Kuperman et al., 2023). Integrating eye-movement features with reader profiles therefore offers a "
            "credible route toward adaptive mediation without requiring continuous human–learner dialogues.\n\n"
            "From a validity standpoint, the proposal treats process data as auxiliary evidence that constrains "
            "interpretation of mediated performance: gaze indices are not proposed as standalone proficiency scores, "
            "but as evidence that helps explain why particular mediations are warranted for particular learners on "
            "particular items. This distinction matters for programme priorities around fairness and transparency in "
            "digital assessment, where claims about “emerging abilities” should be traceable to observable behaviours "
            "and principled mediation rules (Poehner & Wang, 2021)."
        ),
    ),
    (
        "Research gap",
        (
            "A critical gap separates the scalability of Interventionist C-DA from the responsiveness of "
            "Interactionist DA. Current platforms rarely use continuous process evidence to adapt mediation; cues "
            "typically follow predetermined ladders. Parallel gaps exist between product-only scoring and "
            "construct-relevant process data that can explain why a response is (in)correct for a given learner.\n\n"
            "This project proposes Adaptive Interventionist Dynamic Assessment (A-IDA): a synthesis in which "
            "eye-tracking metrics and reader-profile classifications jointly trigger individualised scaffolding "
            "during computerised reading tests. The objective is to approximate the diagnostic richness of "
            "interactionist mediation within a scalable workflow suitable for empirical comparison against "
            "traditional C-DA and non-mediated control conditions."
        ),
    ),
    (
        "Research questions",
        (
            "1. Are there significant differences in reading comprehension outcomes and response times among "
            "participants assigned to A-IDA, a traditional computerised Interventionist DA sequence, and a "
            "non-mediated control condition?\n\n"
            "2. How do comprehension outcomes and response times vary across reader profiles (e.g., highly competent, "
            "effortful, challenged) under A-IDA compared with traditional C-DA?\n\n"
            "3. What relationships obtain between eye-movement parameters (e.g., fixation measures, regressive "
            "saccades) and the type and frequency of scaffolding cues triggered during testing?"
        ),
    ),
    (
        "Research design",
        (
            "The study follows design-based research (DBR) using the ADDIE cycle (Analysis, Design, Development, "
            "Implementation, Evaluation) to build, pilot, and refine A-IDA across iterative micro-cycles. DBR is "
            "appropriate because the artefact (an adaptive testing interface) and its instrumentation are co-developed "
            "with evidence from authentic learner use (McNamara, 2014; Poehner & Wang, 2021).\n\n"
            "Figure 1 summarises the processing architecture. In live operation, gaze streams are segmented into "
            "fixations and saccades; features feed a profile-aware decision layer that selects mediation from a "
            "validated bank mapped to item-level skills (e.g., inferencing, cohesion). This figure should appear "
            "as a simple flowchart in the final Word file (Insert → SmartArt → Process, or an exported diagram from "
            "your design tool).\n\n"
            "Table 1 outlines a 12-month timeline aligned to the programme window (December 2026–December 2027). "
            "Months are indicative; exact dates will be anchored to institutional calendars and ethics approval."
        ),
    ),
]

for heading, body in sections:
    doc.add_heading(heading, level=1)
    doc.add_paragraph(body)

# Table 1
doc.add_paragraph()
t = doc.add_table(rows=1, cols=4)
hdr = t.rows[0].cells
hdr[0].text = "Phase (ADDIE)"
hdr[1].text = "Period"
hdr[2].text = "Primary outputs"
hdr[3].text = "Evaluation focus"

rows = [
    ("Analysis", "Dec 2026 – Jan 2027", "Needs analysis; risk register; mediation taxonomy draft", "Construct map ↔ Cambridge item types"),
    ("Design", "Feb – Mar 2027", "Wireframes; cue database v1; pilot items; preregistration plan", "Expert panel review (n = 5)"),
    ("Development", "Apr – Jul 2027", "EyeLink–PsychoPy/PyQt integration; real-time feature pipeline; classifier v1", "Latency benchmarks; debugging logs"),
    ("Implementation", "Aug – Sep 2027", "Pilot (n ≈ 20–30); refine mediation rules", "Usability; mediation validity checks"),
    ("Evaluation", "Oct – Dec 2027", "Main study; analysis; report; dissemination package", "Effect sizes; process–outcome models"),
]
for phase, period, outputs, ev in rows:
    r = t.add_row().cells
    r[0].text = phase
    r[1].text = period
    r[2].text = outputs
    r[3].text = ev

doc.add_paragraph()
doc.add_paragraph("Figure 1. A-IDA live cycle: gaze acquisition → event detection (fixations, regressions) → "
                  "profile-aware mediation selection → delivery via the testing UI → logged outcomes for analysis.")

doc.add_heading("Methodology", level=1)

method_parts = [
    (
        "Rationale for proposed methods",
        (
            "Eye-tracking is included because it captures how comprehension unfolds, not only whether an item is "
            "answered correctly (Godfroid, Winke, & Conklin, 2020). Fixation- and regression-based features provide "
            "proxies for difficulty and strategic reanalysis that can be computed within trial windows compatible "
            "with mediated items. Profile classification (Alexander, 2005; Dinsmore et al., 2018) supplies stable "
            "priors that prevent overfitting idiosyncratic noise in gaze data. The learning potential score (LPS) "
            "framework remains appropriate because it quantifies responsiveness to mediation relative to "
            "unassisted performance (Poehner et al., 2015). Finally, scikit-learn decision rules offer an "
            "interpretable first-generation mapping from features to mediation levels—transparent to auditors and "
            "teachers—while remaining fast enough for classroom/lab deployment; more complex models can be explored "
            "in later iterations once trace data are logged."
        ),
    ),
    (
        "Participants and research team",
        (
            "Participants will be EFL undergraduates enrolled in English-medium coursework at a public university "
            "in Türkiye. A vocabulary-size screening will target intermediate–upper intermediate readers "
            "(approximately 5,000–6,000 word families) to align texts with Cambridge-level tasks. Reader profiles "
            "will be operationalised using the Alexander (2005) competence framework as informed by interest/prior "
            "knowledge indicators (Dinsmore et al., 2018). Target recruitment for the main phase is approximately "
            "60–90 participants (20–30 per arm), with a preceding pilot to stabilise mediation thresholds. The team "
            "includes applied linguists/assessment researchers and a software engineer responsible for the adaptive "
            "interface, aligning with Jin and Liu’s (2024) emphasis on principled mediation design in computerised DA."
        ),
    ),
    (
        "Instruments and data collection",
        (
            "Reading materials and items will be drawn from validated Cambridge English examinations (e.g., B2 First, "
            "C1 Advanced, or IELTS Academic reading), ensuring construct alignment with Cambridge constructs and score "
            "interpretations. An expert panel (five applied linguists) will review text–mediation coupling before pilots.\n\n"
            "Eye-tracking will use the SR Research EyeLink Portable Duo already available in the lab. The software stack "
            "(pylink, PsychoPy, PyQt5; NumPy/Pandas for online feature extraction) prioritises timing fidelity and "
            "reproducibility. Mediation banks will follow graduated prompting consistent with Interventionist DA while "
            "allowing profile-conditioned branches—an implementation parallel to hybrid C-DA rationales (Jin & Liu, 2024)."
        ),
    ),
    (
        "Procedure and analysis",
        (
            "Participants will be randomly assigned to A-IDA, standard C-DA, or control. Dependent variables include "
            "accuracy, response time, mediation level required, and LPS contrasts. Omnibus tests (e.g., one-way ANOVA "
            "or robust alternatives if assumptions fail) will compare arms; exploratory moderation by reader profile "
            "will use factorial or regression models; relationships between gaze features and mediation will be "
            "examined via correlational and regression frameworks, with multiple-comparison control as appropriate.\n\n"
            "Ethics: institutional ethics approval, informed consent, right to withdraw, secure storage of gaze and "
            "behavioural logs, and clear protocols for incidental findings. Data management will follow least-privilege "
            "access and pseudonymised identifiers.\n\n"
            "Limitations and delimitations: eye-tracking requires laboratory-quality calibration; findings will "
            "generalise first to similar proficiency bands and task types rather than all Cambridge products. "
            "Mediation rules will be versioned (A-IDA v1) to support replication. Finally, while the decision layer "
            "begins with interpretable rules, future work may explore richer sequence models under strict governance "
            "for high-stakes use.\n\n"
            "Transparency outputs: alongside peer-targeted dissemination, the team will prepare a mediation codebook "
            "(cue functions, triggering thresholds, and profile logic), anonymised analysis scripts, and a summary of "
            "alignment between gaze features and item-level skills. Where copyright permits, de-identified process "
            "aggregates may be shared to support secondary analysis; otherwise, synthetic illustrations will document "
            "the approach for Cambridge stakeholders."
        ),
    ),
]

for sub, body in method_parts:
    doc.add_heading(sub, level=2)
    doc.add_paragraph(body)

doc.add_heading("Potential implications for Cambridge University Press & Assessment and the wider field", level=1)
doc.add_paragraph(
    "For Cambridge English, the project offers a research pathway toward digital assessments and learning products "
    "that combine high-quality items with transparent, profile-sensitive feedback—supporting fairness arguments where "
    "learners differ in processing signatures yet demonstrate similar emerging abilities when appropriately mediated. "
    "Because materials come from Cambridge examinations, results can speak directly to how diverse profiles engage "
    "with existing tasks, informing future item development and automated tutoring overlays. If process-sensitive "
    "mediation improves score meaning without inflating construct-irrelevant variance, the model also aligns with "
    "programme interests in assessment for learning and responsible innovation in computer-delivered testing.\n\n"
    "For the wider field, the work contributes an empirically grounded synthesis of DA, eye-tracking process data, "
    "and adaptive testing. It demonstrates how “emerging construct” claims can be evidenced through linked process–"
    "product modelling rather than outcome-only inference (cf. Kuperman et al., 2023; Poehner & Wang, 2021). "
    "Finally, documenting mediation rules and gaze-derived features advances open, auditable pathways for inclusive "
    "assessment design beyond one-size-fits-all adaptive algorithms, including potential transfer of cueing logic to "
    "settings where specialised hardware is unavailable (e.g., using simplified behavioural proxies in standard "
    "computer-based tests)."
)

doc.add_heading("References", level=1)

refs = [
    "Alexander, P. A. (2005). The path to competence: A lifespan developmental perspective on reading. "
    "Journal of Literacy Research, 37(4), 413–436.",
    "Dinsmore, D. L., Fox, E., Parkinson, M. M., & Bilgili, F. (2018). The interplay of prior knowledge, interest, "
    "and reading comprehension. Learning and Individual Differences, 65, 477–490.",
    "Godfroid, A., Winke, P., & Conklin, K. (2020). Exploring the depths of second language processing with eye "
    "tracking: An introduction. Second Language Research, 36(3), 243–256. https://doi.org/10.1177/0267658320922578",
    "Jin, C., & Liu, Y. (2024). Diagnosing and promoting learners’ L2 inferential reading development through hybrid "
    "computerised dynamic assessment in the Chinese EFL classroom. Computer Assisted Language Learning. "
    "Advance online publication. https://doi.org/10.1080/09588221.2024.2421521",
    "Kozulin, A., & Garb, E. (2002). Dynamic assessment of EFL text comprehension. School Psychology International, "
    "23(1), 112–127.",
    "Kuperman, V., Siegelman, N., Schroeder, S., Acartürk, C., Alexeeva, S., Amenta, S., Bertram, R., Bonandrini, R., "
    "Brysbaert, M., Chernova, D., Da Fonseca, S. M., Dirix, N., Duyck, W., Fella, A., Frost, R., Gattei, C. A., "
    "Kalaitzi, A., Lõo, K., Marelli, M., … Usal, K. A. (2023). Text reading in English as a second language: "
    "Evidence from the Multilingual Eye-Movements Corpus. Studies in Second Language Acquisition, 45(1), 3–37. "
    "https://doi.org/10.1017/S0272263121000954",
    "Lantolf, J. P., & Poehner, M. E. (2004). Dynamic assessment of L2 development: Bringing the past into the future. "
    "Journal of Applied Linguistics, 1(1), 49–72.",
    "Lantolf, J. P., & Poehner, M. E. (2010). Dynamic assessment in the classroom: Vygotskian praxis for second language "
    "development. Language Teaching Research, 15(1), 11–33.",
    "McNamara, T. (2014). 30 years on—Evolution or revolution? Language Assessment Quarterly, 11(2), 226–232.",
    "Poehner, M. E., & Lantolf, J. P. (2005). Dynamic assessment in the language classroom. Language Teaching Research, "
    "9(3), 233–265.",
    "Poehner, M. E., & Lantolf, J. P. (2013). Bringing the ZPD into the equation: Capturing L2 development during "
    "Computerized Dynamic Assessment (C-DA). Language Teaching Research, 17(3), 323–342.",
    "Poehner, M. E., & Wang, Z. (2021). Dynamic Assessment and second language development. Language Teaching, 54(4), "
    "472–490. https://doi.org/10.1017/S0261444820000555",
    "Poehner, M. E., Zhang, J., & Lu, X. (2015). Computerized dynamic assessment (C-DA): Diagnosing L2 development "
    "according to learner responsiveness to mediation. Language Testing, 32(3), 337–357.",
    "Vygotsky, L. S. (1978). Mind in society: The development of higher psychological processes. Harvard University Press.",
]

for r in refs:
    p = doc.add_paragraph(r)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.first_line_indent = Pt(-18)

doc.save(OUT)
print("Wrote", OUT)
