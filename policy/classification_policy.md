# Classification Policy — Green vs Red Documents

## Purpose of This Policy

This policy defines how documents should be classified into Green (non‑sensitive) or Red (sensitive) for the purpose of this proof‑of‑concept. The rules are intentionally simple so an **LLM** can apply them reliably.

## Green Documents (Store in G/ folder)

Green documents are safe to store in the cloud and safe to make public. They contain no sensitive information about individuals, customers, or internal operations.

A document is Green if it meets all of the following:

- It contains general information intended for the public

- It does not contain personal data

- It does not contain customer‑specific details

- It does not contain signatures, IDs, or financial information

- It does not describe internal processes that could be misused

- It is informational, educational, or promotional in nature

## Examples of Green documents

- Public brochures

- General information sheets

- Environmental or water‑usage guides

- Public safety information

- Marketing material

- High‑level descriptions of services

- Public forms that contain no filled‑in data

- If the document looks like something that could be posted on a public website, it is Green.

## Red Documents (Store in R/ folder)

Red documents contain sensitive information and must not be stored in the cloud. They must remain on‑premises.

A document is Red if it contains any of the following:

## Personal Data

- Names

- Addresses

- Phone numbers

- Email addresses

- Personal identity numbers (personnummer)

- Customer account numbers

- Signatures

## Customer‑Specific Information

- Filled‑in forms

- Complaints

- Damage reports

- Ownership change forms

- Water meter replacement forms

- Any document tied to a specific person or property

## Sensitive Operational Information

- Internal procedures not meant for the public

- Technical details that could expose infrastructure

- Internal project documents

- Internal risk assessments

## Financial or Contractual Information

- Invoices

- Payment details

- Bank information

- Contracts or agreements

- If the document contains anything that identifies a person, property, or internal process, it is Red.

Simple Decision Rule (**LLM**‑Friendly) If the document contains personal data or customer‑specific information → Red. If it is general information intended for the public → Green.

### Edge Cases

If the **LLM** is unsure:

- Default to Red

Red is always the safer choice

## Output Format for the Classifier

When classifying a document, the **LLM** should output:

```json
{
_classification_: _Green_ or _Red_,
_reason_: _Short explanation in plain English_
}
```
