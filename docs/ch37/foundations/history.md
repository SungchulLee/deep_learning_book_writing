# Historical Context and Landmark Cases

## Overview

The study of algorithmic fairness did not emerge in a vacuum. It builds on decades of anti-discrimination law, statistical testing in employment, and a series of high-profile cases where ML systems produced demonstrably unfair outcomes. Understanding this history provides essential context for the mathematical definitions and mitigation techniques that follow.

## Legal Foundations

### Disparate Treatment and Disparate Impact

U.S. anti-discrimination law distinguishes two forms of discrimination:

**Disparate treatment** occurs when decisions are explicitly based on a protected attribute. For example, a lending policy that sets different interest rates based on race constitutes disparate treatment.

**Disparate impact** (established in *Griggs v. Duke Power Co.*, 1971) occurs when a facially neutral policy disproportionately affects a protected group. A hiring test that is equally administered to all applicants but has a pass rate that differs significantly across racial groups may constitute disparate impact, even absent discriminatory intent.

The **80% rule** (EEOC guidelines) formalizes disparate impact: a selection rate for any group that is less than 80% (four-fifths) of the rate for the group with the highest selection rate constitutes evidence of adverse impact:

$$\text{Disparate Impact Ratio} = \frac{\text{Selection Rate}_{\text{minority}}}{\text{Selection Rate}_{\text{majority}}} \geq 0.8$$

### Key Legislation

| Legislation | Year | Domain | Relevance to ML |
|------------|------|--------|-----------------|
| Civil Rights Act (Title VII) | 1964 | Employment | Prohibits discrimination in hiring |
| Equal Credit Opportunity Act (ECOA) | 1974 | Lending | Prohibits credit discrimination |
| Fair Housing Act | 1968 | Housing | Prohibits discrimination in housing |
| GDPR (EU) | 2018 | General | Right to explanation, non-discrimination |
| EU AI Act | 2024 | General | Risk-based regulation of AI systems |

## Landmark Cases in Algorithmic Fairness

### COMPAS Recidivism Prediction (2016)

The COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) system, developed by Northpointe (now Equivant), assessed the likelihood of criminal recidivism. ProPublica's investigation revealed that the tool exhibited significant racial disparities:

- Black defendants were nearly twice as likely as white defendants to be falsely flagged as high-risk (higher FPR)
- White defendants were more likely to be incorrectly labeled low-risk when they actually did reoffend (higher FNR)

Northpointe responded that COMPAS was *calibrated*: among defendants scored as high-risk, approximately equal proportions of Black and white defendants actually reoffended. This dispute crystallized a fundamental insight: **calibration and equal error rates cannot simultaneously hold when base rates differ across groups** (see [Chouldechova's Theorem](../impossibility/chouldechova.md)).

### Amazon Hiring Tool (2018)

Amazon developed an AI recruiting tool trained on historical hiring data from a male-dominated industry. The system learned to penalize résumés containing the word "women's" (as in "women's chess club") and downgraded graduates of all-women's colleges. The system was scrapped before deployment, but it illustrated how historical bias in training data perpetuates discrimination.

### Apple Card Credit Limits (2019)

Apple's credit card, underwritten by Goldman Sachs, was investigated after reports that it offered significantly lower credit limits to women than to men with comparable financial profiles. The case highlighted how even without using gender as an explicit input, algorithms can produce discriminatory outcomes through proxy variables and historical patterns in financial data.

### Healthcare Algorithm Bias (2019)

Obermeyer et al. (2019) discovered that a widely used healthcare algorithm exhibited significant racial bias. The system used healthcare costs as a proxy for healthcare needs. Because Black patients historically had less access to healthcare (and thus incurred lower costs), the algorithm systematically underestimated their health needs. At a given risk score, Black patients were significantly sicker than white patients with the same score.

## Evolution of Fairness Research

### Phase 1: Statistical Testing (1960s–1990s)

Early fairness work focused on employment testing and the statistical validation of selection procedures. The four-fifths rule and methods for detecting adverse impact were developed in this era.

### Phase 2: Formalization (2010s)

The ML fairness field crystallized around formal definitions:

- **Dwork et al. (2012)**: "Fairness Through Awareness"—individual fairness via Lipschitz constraints
- **Hardt, Price, & Srebro (2016)**: Equalized odds and equal opportunity
- **Chouldechova (2017)**: Impossibility of simultaneous calibration and equal error rates
- **Kleinberg, Mullainathan, & Raghavan (2016)**: KMR impossibility theorem

### Phase 3: Mitigation and Practice (2017–present)

Research shifted toward practical mitigation:

- Pre-processing methods (Calmon et al., 2017; Kamiran & Calders, 2012)
- In-processing methods (Zhang et al., 2018; Agarwal et al., 2018)
- Post-processing methods (Hardt et al., 2016; Pleiss et al., 2017)
- Fairness toolkits: IBM AI Fairness 360, Google What-If Tool, Microsoft Fairlearn

### Phase 4: Regulation and Governance (2020s)

Governments have begun to regulate algorithmic decision-making:

- EU AI Act classifies AI systems by risk level with mandatory fairness requirements for high-risk applications
- U.S. executive orders on AI safety and equity
- Financial regulators (OCC, Fed, CFPB) issuing guidance on model risk management with fairness components

## Implications for Quantitative Finance

Financial applications face particularly stringent fairness requirements:

1. **Regulatory mandates**: ECOA, Fair Housing Act, and fair lending regulations explicitly prohibit discrimination
2. **Disparate impact liability**: Financial institutions can be held liable for algorithmic disparate impact even without intent
3. **Model risk management**: SR 11-7 (OCC/Fed) requires documentation and validation of model fairness
4. **Reputational risk**: Public revelation of algorithmic bias carries significant reputational and financial consequences

```python
def timeline_summary():
    """Print a summary timeline of key fairness milestones."""
    milestones = [
        (1964, "Civil Rights Act (Title VII)"),
        (1971, "Griggs v. Duke Power — disparate impact doctrine"),
        (1974, "Equal Credit Opportunity Act"),
        (1978, "EEOC Uniform Guidelines — four-fifths rule"),
        (2012, "Dwork et al. — Fairness Through Awareness"),
        (2016, "ProPublica COMPAS investigation"),
        (2016, "Hardt et al. — Equalized Odds / Equal Opportunity"),
        (2016, "Kleinberg et al. — KMR Impossibility Theorem"),
        (2017, "Chouldechova — Calibration impossibility"),
        (2017, "IBM AI Fairness 360 toolkit released"),
        (2018, "Amazon hiring tool bias revealed"),
        (2018, "Zhang et al. — Adversarial Debiasing"),
        (2019, "Apple Card gender bias investigation"),
        (2019, "Obermeyer et al. — Healthcare algorithm bias"),
        (2020, "Microsoft Fairlearn 0.5 released"),
        (2024, "EU AI Act enters into force"),
    ]
    
    print("Key Milestones in Algorithmic Fairness")
    print("=" * 55)
    for year, event in milestones:
        print(f"  {year}  {event}")

timeline_summary()
```

## Key Takeaways

1. **Legal frameworks** for anti-discrimination predate ML but apply directly to algorithmic decisions
2. **Landmark cases** (COMPAS, Amazon, Apple Card) have driven public and academic attention to fairness
3. **The COMPAS debate** directly motivated the impossibility theorems covered in §36.3
4. **Financial regulation** imposes particularly strict fairness requirements on ML models in lending, insurance, and trading
5. **The field is rapidly evolving**, with new regulations and tools emerging regularly

## Next Steps

- [Demographic Parity](../definitions/demographic_parity.md): The first and simplest group fairness definition
- [Chouldechova's Theorem](../impossibility/chouldechova.md): Why the COMPAS debate was mathematically inevitable
