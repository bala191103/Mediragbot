# dashboard metrices

import os
import pandas as pd
import streamlit as st
import json
from datetime import datetime

def render_metrics_dashboard(logs_session_state):
    # Add spacing at top of metrics tab
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<h1 class="section-header">📊 RAG Evaluation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Real-time performance metrics and evaluation data for the MediRAG system.")

    # Add spacing after description
    st.markdown("<br>", unsafe_allow_html=True)

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/metrics.csv"

    # Load and merge data
    if os.path.exists(csv_path):
        try:
            df_file = pd.read_csv(csv_path)
            if logs_session_state:
                df_mem = pd.DataFrame(logs_session_state)
                df = pd.concat([df_file, df_mem]).drop_duplicates(subset=["time", "question"], keep="last")
            else:
                df = df_file
        except Exception:
            df = pd.DataFrame(logs_session_state)
    else:
        df = pd.DataFrame(logs_session_state)

    if df.empty:
        # Enhanced empty state with better spacing
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.info("No evaluation data available yet. Start chatting to generate metrics and insights!")
        st.markdown("""
        ### What you'll see here:
        - Faithfulness Score: How well responses stick to the uploaded documents
        - Answer Relevancy: How well responses address user questions  
        - Context Relevancy: How relevant retrieved context is to queries
        - Response Times: System performance metrics
        - User Feedback: Real-time satisfaction tracking
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Convert numeric columns
        for col in ["faithfulness", "answer_relevancy", "context_relevancy", "gen_latency_sec", "eval_latency_sec"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Summary metrics with enhanced spacing
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Performance Overview")
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4, gap="large")

        with col1:
            faith_avg = df['faithfulness'].mean() if 'faithfulness' in df.columns else 0
            st.metric("Faithfulness", f"{faith_avg:.3f}", help="How well responses stick to document content")

        with col2:
            ans_avg = df['answer_relevancy'].mean() if 'answer_relevancy' in df.columns else 0
            st.metric("Answer Relevancy", f"{ans_avg:.3f}", help="How well responses address user questions")

        with col3:
            ctx_avg = df['context_relevancy'].mean() if 'context_relevancy' in df.columns else 0
            st.metric("Context Relevancy", f"{ctx_avg:.3f}", help="How relevant retrieved content is")

        with col4:
            latency_avg = df['gen_latency_sec'].mean() if 'gen_latency_sec' in df.columns else 0
            st.metric("Avg Response Time", f"{latency_avg:.2f}s", help="Average time to generate responses")

        st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing between sections
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Interactive data table with enhanced styling
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Recent Interactions")
        st.markdown("<br>", unsafe_allow_html=True)

        show_cols = ["time", "question", "faithfulness", "answer_relevancy", "context_relevancy", "gen_latency_sec", "feedback"]
        available_cols = [col for col in show_cols if col in df.columns]

        if available_cols:
            display_df = df[available_cols].tail(10).round(3)
            st.dataframe(display_df, use_container_width=True, height=350)

        st.markdown('</div>', unsafe_allow_html=True)
        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Performance trends with enhanced styling
        if {"faithfulness", "answer_relevancy", "context_relevancy"}.issubset(df.columns):
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("### Performance Trends")
            st.markdown("<br>", unsafe_allow_html=True)
            chart_data = df[["faithfulness", "answer_relevancy", "context_relevancy"]].tail(20)
            st.line_chart(chart_data, height=400)
            st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Detailed analysis with better layout
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            if "gen_latency_sec" in df.columns:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown("### Response Time Distribution")
                st.markdown("<br>", unsafe_allow_html=True)
                st.bar_chart(df["gen_latency_sec"].tail(10), height=300)
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if "feedback" in df.columns:
                feedback_counts = df["feedback"].value_counts()
                if not feedback_counts.empty:
                    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                    st.markdown("### User Feedback Summary")
                    st.markdown("<br>", unsafe_allow_html=True)
                    for feedback, count in feedback_counts.items():
                        if feedback and feedback != "None":
                            st.write(f"{feedback}: {count} responses")
                    st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Judge reasoning analysis with enhanced styling
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### AI Judge Analysis (Last 5 Interactions)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        if "judge_reasons" in df.columns:
            for idx, (_, row) in enumerate(df.tail(5).iterrows()):
                with st.expander(f"Query {idx + 1}: {row.get('question', '')[:80]}..."):
                    try:
                        reasons = json.loads(row.get("judge_reasons", "{}"))
                        for dimension, reason in reasons.items():
                            if reason:
                                st.write(f"{dimension.title()}: {reason}")
                    except Exception:
                        st.write("Raw judge data:", row.get("judge_reasons", "No data"))
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Add spacing
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Export functionality with enhanced styling
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Data Export")
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            if st.button("Download CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Complete Data",
                    data=csv,
                    file_name=f"medirag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("Clear All Data", use_container_width=True):
                # Add a confirmation dialog
                if st.button("⚠ Confirm Clear All", use_container_width=True):
                    st.session_state.logs = []
                    if os.path.exists(csv_path):
                        os.remove(csv_path)
                    st.success("All data cleared!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
