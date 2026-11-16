
import streamlit as st


def display_report(
    answers,
    contradictions,
    score_result
):
    st.header("Interview Analysis Report")

    st.subheader("Overall Risk Assessment")

    st.metric(
        label="Experimental Risk Indicator",
        value=f"{score_result['score']}%"
    )

    st.write(f"**Assessment:** {score_result['label']}")

    st.warning(
        "This is an experimental risk estimate, not proof "
        "that a person is lying. Inconsistencies can occur "
        "for many reasons, including memory errors."
    )

    st.divider()

    st.subheader("Answer Summary")

    for question, answer in answers.items():
        with st.expander(question.replace("_", " ").title()):
            st.write(answer)

    st.divider()

    st.subheader("Detected Inconsistencies")

    if contradictions:
        for item in contradictions:
            st.warning(item)
    else:
        st.success("No contradictions detected by the current rules.")

    st.divider()

    st.subheader("Score Breakdown")

    st.write(
        f"Contradiction indicator: "
        f"{score_result['contradiction_score']}%"
    )

    st.write(
        f"Voice deviation indicator: "
        f"{score_result['voice_deviation']}%"
    )

    st.write(
        f"Existing model score: "
        f"{score_result['model_score']}%"
    )
