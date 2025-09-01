"""
Streamlit prototype for insulin pump education
------------------------------------------------

This module defines a simple Streamlit application aimed at
educating medical students, residents and fellows about the
principles of insulin pump therapy.  It simulates a cohort of
ten synthetic patients with one year of continuous glucose
monitoring data, insulin pump settings and clinical notes.  The
app presents these data longitudinally, allowing the learner to
select a patient, scroll through each month of the year and
observe trends in blood glucose, basal insulin delivery and
bolus dosing.  Contextual notes describe important events such
as adjustments for dawn phenomenon, exercise or illness, and
recommendations are generated automatically based on the patient
and month.  The goal is to provide an interactive journey
rather than a series of static multiple‑choice questions so
learners can develop a feel for real‑world insulin pump
management.

The synthetic data are not intended to represent any actual
patient.  Values are generated pseudo‑randomly within
physiologically plausible ranges.  Recommendations are simple
heuristics to illustrate how one might adjust insulin doses in
response to trends – they are not clinical advice.

This file can be run with ``streamlit run insulin_pump_app.py``
to launch the app locally.  Once started, the user can select a
patient from the sidebar and use the month slider to move
through the longitudinal case.  Charts update automatically and
notes provide narrative context for each month.  This design
aligns with educational best practices that emphasise active
engagement and narrative learning rather than rote recall
【18†L1-L8】.
"""

from __future__ import annotations

import datetime
import random
import string
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class MonthlyData:
    """Container for a single month's synthetic data.

    Attributes
    ----------
    bg_values: List[float]
        Daily blood glucose readings for the month (mg/dL).
    basal_units: float
        Total basal insulin delivered per day (units/day) –
        representing the mean basal requirement for the month.
    bolus_units: float
        Total bolus insulin delivered per day (units/day).
    notes: str
        Narrative describing events, adjustments or patient
        experiences in the month.
    """

    bg_values: List[float]
    basal_units: float
    bolus_units: float
    notes: str


@dataclass
class Patient:
    """Data structure representing a synthetic insulin pump user."""

    id: int
    name: str
    pump_type: str
    months: Dict[int, MonthlyData] = field(default_factory=dict)

    def average_bg(self, month_index: int) -> float:
        """Return the average blood glucose for a given month."""
        month_data = self.months[month_index]
        return float(np.mean(month_data.bg_values))

    def metrics_for_month(self, month_index: int) -> Tuple[float, float, float]:
        """Return average BG, basal units and bolus units for the month."""
        md = self.months[month_index]
        avg_bg = float(np.mean(md.bg_values))
        return avg_bg, md.basal_units, md.bolus_units


def _generate_name() -> str:
    """Generate a random human‑readable name.

    To make the synthetic patients easier to relate to, this helper
    picks a given name and family name from short lists.  In a
    classroom setting these names reinforce the human dimension
    without referencing real individuals.
    """
    first_names = [
        "Alex", "Casey", "Jordan", "Taylor", "Morgan",
        "Robin", "Blake", "Sydney", "Jamie", "Avery",
    ]
    last_names = [
        "Nguyen", "Smith", "Patel", "Kim", "Garcia",
        "Brown", "Hernandez", "Lee", "Johnson", "Davis",
    ]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def create_patients(num_patients: int = 10) -> List[Patient]:
    """Construct a list of synthetic patients with one year of data.

    Parameters
    ----------
    num_patients: int, default 10
        Number of synthetic patients to generate.

    Returns
    -------
    List[Patient]
        Generated patient objects populated with monthly data.

    Notes
    -----
    Each patient is assigned a pump type (Omnipod, Tandem or
    Medtronic) and receives random but realistic basal and bolus
    insulin requirements.  Blood glucose values are sampled from
    distributions centred around 110–180 mg/dL, reflecting common
    glycaemic targets【39†L5-L13】.  The narrative notes describe
    typical scenarios such as basal adjustments for dawn
    phenomenon【23†L633-L641】, exercise management【37†L9-L17】 or
    sick day increases【38†L19-L25】.
    """
    random.seed(42)
    np.random.seed(42)
    patients: List[Patient] = []
    pump_choices = ["Omnipod", "Tandem t:slim X2", "Medtronic 780G"]
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    for pid in range(num_patients):
        name = _generate_name()
        pump = random.choice(pump_choices)
        patient = Patient(id=pid, name=name, pump_type=pump)
        for m_idx in range(12):
            # Simulate daily blood glucose values for each day of month
            days_in_month = 30  # fixed for simplicity
            # Base mean BG around 140 mg/dL with variation; modify for pump type
            base_mean = 140 + np.random.normal(0, 10)
            # Slightly lower mean if using advanced hybrid pump (Tandem or Medtronic)
            if pump != "Omnipod":
                base_mean -= 5
            # Simulate BG values with normal distribution and clip to plausible range (50–350)
            bg_values = np.random.normal(loc=base_mean, scale=20, size=days_in_month)
            bg_values = np.clip(bg_values, 50, 350).tolist()
            # Basal units/day: around 20–40 units depending on patient size; vary over months
            basal_units = float(np.random.uniform(18, 40))
            # Bolus units/day: meal and correction doses; around 15–35 units
            bolus_units = float(np.random.uniform(15, 35))
            # Compose narrative note based on month and random events
            note = ""
            month_name = month_names[m_idx]
            # Dawn phenomenon adjustments early in the year
            if m_idx in [0, 1] and base_mean > 150:
                note = (
                    f"Observed higher fasting glucose in {month_name}. "
                    "Increased early‑morning basal to address the dawn phenomenon【23†L633-L641】."
                )
            # Exercise adjustments in spring/summer
            elif m_idx in [3, 4, 5] and np.mean(bg_values) < 120:
                note = (
                    f"Regular aerobic exercise in {month_name} lowered glucose levels. "
                    "Temporary basal reduction of 50% during workouts was applied【37†L9-L17】."
                )
            # Sick day high in winter
            elif m_idx in [9] and np.mean(bg_values) > 170:
                note = (
                    f"Illness in {month_name} increased insulin requirements. "
                    "Temporary basal increased by 30% as per sick‑day protocol【38†L19-L25】."
                )
            else:
                note = f"Routine month of pump therapy in {month_name}. Continued monitoring."
            md = MonthlyData(
                bg_values=bg_values,
                basal_units=basal_units,
                bolus_units=bolus_units,
                notes=note,
            )
            patient.months[m_idx] = md
        patients.append(patient)
    return patients


def recommend_adjustment(patient: Patient, month_index: int) -> str:
    """Provide a simple recommendation based on average blood glucose.

    Parameters
    ----------
    patient: Patient
        The patient whose data we are analysing.
    month_index: int
        Index of the month (0–11).

    Returns
    -------
    str
        A narrative recommendation suggesting how to adjust pump settings.

    Notes
    -----
    If average glucose exceeds 180 mg/dL, the function suggests
    increasing basal or bolus dosing; if it is below 80 mg/dL, it
    recommends reducing insulin.  Otherwise, it affirms that
    settings are appropriate.  These heuristics are simplistic and
    serve illustrative purposes only.
    """
    avg_bg, basal_units, bolus_units = patient.metrics_for_month(month_index)
    recommendation: str
    if avg_bg > 180:
        recommendation = (
            "Average blood glucose is above target. Consider increasing basal "
            "rates in the early morning or adjusting the insulin‑to‑carbohydrate ratio "
            "for meals. Review bolus timing to ensure pre‑meal dosing."
        )
    elif avg_bg < 80:
        recommendation = (
            "Average blood glucose is below target. Reduce basal delivery or increase "
            "carbohydrate intake at snacks. Consider setting a temporary basal decrease "
            "during times of increased activity."
        )
    else:
        recommendation = (
            "Average blood glucose is within target range. Continue current pump settings "
            "while maintaining regular monitoring."
        )
    return recommendation


def _format_month(index: int) -> str:
    """Return the human‑readable month name for the given index."""
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return month_names[index]


def main() -> None:
    """Run the Streamlit app."""
    st.set_page_config(page_title="Insulin Pump Education", layout="wide")
    st.title("Insulin Pump Therapy: Longitudinal Case Studies")
    st.markdown(
        """
        This educational prototype allows you to explore the one‑year journey of
        several synthetic patients on insulin pump therapy.  Select a patient
        from the sidebar and use the month slider to step through their
        experience.  Charts display trends in blood glucose and insulin
        delivery, and narrative notes explain key events such as basal
        adjustments for the dawn phenomenon【23†L633-L641】, exercise adaptations【37†L9-L17】
        and sick‑day protocols【38†L19-L25】.  Recommendations are generated to
        illustrate how clinicians might respond to observed patterns.  These
        simulations are for training purposes only and do not reflect real
        patient data.
        """
    )
    # Sidebar controls
    patients = create_patients(10)
    patient_names = [p.name for p in patients]
    selected_name = st.sidebar.selectbox("Select patient", patient_names)
    selected_patient = next(p for p in patients if p.name == selected_name)
    month_index = st.sidebar.slider(
        "Month", min_value=0, max_value=11, value=0, format="%d",
        help="Scroll through the months of the year to see changes over time.",
    )
    # Display high‑level patient information
    st.subheader(f"Patient profile: {selected_patient.name}")
    st.write(f"**Pump type:** {selected_patient.pump_type}")
    # Monthly summary
    month_name = _format_month(month_index)
    st.markdown(f"### {month_name}")
    avg_bg, basal_units, bolus_units = selected_patient.metrics_for_month(month_index)
    # Create charts in columns
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.markdown("**Blood Glucose Trend**")
        md = selected_patient.months[month_index]
        bg_df = pd.DataFrame({
            "Day": list(range(1, len(md.bg_values) + 1)),
            "Blood Glucose (mg/dL)": md.bg_values,
        })
        st.line_chart(bg_df.set_index("Day"))
    with col2:
        st.markdown("**Insulin Delivery**")
        insulin_df = pd.DataFrame({
            "Type": ["Basal", "Bolus"],
            "Units/day": [md.basal_units, md.bolus_units],
        })
        # Represent basal and bolus as a bar chart
        st.bar_chart(insulin_df.set_index("Type"))
    with col3:
        st.markdown("**Monthly Metrics**")
        st.metric(label="Avg BG (mg/dL)", value=f"{avg_bg:.1f}")
        st.metric(label="Basal units/day", value=f"{basal_units:.1f}")
        st.metric(label="Bolus units/day", value=f"{bolus_units:.1f}")
    # Narrative and recommendations
    st.markdown("### Monthly Narrative")
    st.write(selected_patient.months[month_index].notes)
    st.markdown("### Recommendation")
    rec = recommend_adjustment(selected_patient, month_index)
    st.write(rec)


if __name__ == "__main__":
    main()