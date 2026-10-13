import os
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
# pyrefly: ignore [missing-import]
import pypdf    

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages 2 to end)
        if self._pageNumber > 1:
            self.drawString(36, 810, "CS3103 Machine Learning Lab | Exp-06: Regression for FBG Tilt Prediction")
            self.drawRightString(559, 810, "Student: Mohith Chandra (2403CS04)")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(36, 804, 559, 804)
            
        # Footer
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(36, 35, 559, 35)
        self.drawString(36, 24, "Laboratory Experiment Report – Max 6 Pages Submission")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(559, 24, page_text)
        self.restoreState()

def create_report(output_filename="2403CS04_Lab06_Report.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=42
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1a237e"),
        alignment=1, # Center
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#283593"),
        alignment=1,
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#37474f"),
        alignment=1,
        spaceAfter=8
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0d47a1"),
        spaceBefore=4,
        spaceAfter=3,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1565c0"),
        spaceBefore=3,
        spaceAfter=2,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#212121"),
        spaceAfter=3
    )
    
    bold_body = ParagraphStyle(
        'BoldBody',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.5,
        leading=8,
        alignment=0
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
        alignment=0
    )

    story = []
    
    # =========================================================================
    # PAGE 1: TITLE, META, AIM, DATASET EXPLORATION (TASK 1)
    # =========================================================================
    story.append(Paragraph("INDIAN INSTITUTE OF TECHNOLOGY PATNA", subtitle_style))
    story.append(Paragraph("CS3103: MACHINE LEARNING LABORATORY | EXPERIMENT 06", subtitle_style))
    story.append(Paragraph("Implementation and Comparative Performance Analysis of Traditional Statistical and Conventional Machine Learning Regression Models for FBG-Based Tilt Prediction", title_style))
    story.append(Paragraph("<b>Student Name:</b> Mohith Chandra &nbsp;&nbsp;|&nbsp;&nbsp; <b>Roll Number:</b> 2403CS04 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> October 13, 2026 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Total Marks:</b> 100", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0d47a1"), spaceAfter=5))
    
    story.append(Paragraph("1. Aim and Problem Formulation", h1_style))
    aim_text = (
        "<b>Aim:</b> To design, implement, and rigorously benchmark traditional statistical regression models and conventional "
        "machine learning regression algorithms to estimate biaxial tilt angles (&theta;) using Fiber Bragg Grating (FBG) optical sensor measurements. "
        "The regression task is formulated as mapping input feature vector <b>X</b> = [&Delta;&lambda;<sub>1</sub>, &Delta;&lambda;<sub>2</sub>] &in; &real;<sup>2</sup> "
        "to tilt angle <i>Y</i> = &theta; &in; &real; (axial and transverse tilt analyzed independently). "
        "Validation enforces 5-fold Grouped Cross-Validation grouped strictly on tilt angle, ensuring zero data leakage across cycles."
    )
    story.append(Paragraph(aim_text, body_style))
    
    story.append(Paragraph("2. Task 1: Dataset Exploration and Statistical Analysis", h1_style))
    task1_intro = (
        "The experimental dataset comprises two primary experimental worksheets: <i>Experiment axial</i> and <i>Experiment transverse</i>. "
        "Each sheet contains exactly <b>37 observations</b> recorded over tilt angles from -90&deg; to +90&deg; with a 10&deg; step size. "
        "The first row represents physical sensor measurements rather than textual headers. "
        "Data verification confirmed <b>0 missing values</b> and <b>0 corrupted records</b> across all 7 principal features."
    )
    story.append(Paragraph(task1_intro, body_style))
    
    # Statistical table
    axial_stat_data = [
        [Paragraph(h, table_header) for h in ["Feature", "Description", "Mean", "Median", "Std Dev", "Variance", "Min", "Max"]],
        [Paragraph("tilt_angle", table_cell), Paragraph("Target Angle (&deg;)", table_cell), "0.00", "0.00", "51.85", "2688.89", "-90.00", "90.00"],
        [Paragraph("fbg1", table_cell), Paragraph("FBG1 Peak (nm)", table_cell), "1535.21", "1535.22", "0.023", "0.0005", "1535.18", "1535.25"],
        [Paragraph("fbg2", table_cell), Paragraph("FBG2 Peak (nm)", table_cell), "1547.35", "1547.35", "0.023", "0.0005", "1547.31", "1547.39"],
        [Paragraph("d_lambda1", table_cell), Paragraph("&Delta;&lambda;<sub>1</sub> (pm)", table_cell), "-0.27", "0.00", "23.36", "545.76", "-38.00", "32.00"],
        [Paragraph("d_lambda2", table_cell), Paragraph("&Delta;&lambda;<sub>2</sub> (pm)", table_cell), "-0.86", "-2.00", "23.23", "539.45", "-37.00", "34.00"],
        [Paragraph("d_lambda_diff", table_cell), Paragraph("&Delta;&lambda;<sub>diff</sub> (pm)", table_cell), "0.59", "0.00", "46.46", "2158.42", "-71.00", "68.00"],
        [Paragraph("d_lambda_avg", table_cell), Paragraph("&Delta;&lambda;<sub>avg</sub> (pm)", table_cell), "-0.57", "-0.50", "0.68", "0.46", "-3.00", "0.50"]
    ]
    t_stat = Table(axial_stat_data, colWidths=[65, 95, 45, 45, 48, 52, 45, 45])
    t_stat.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_stat)
    story.append(Spacer(1, 4))
    
    # Task 1 Distribution Image
    story.append(Paragraph("<b>Figure 1: Feature and Angle Probability Distributions (Axial vs Transverse)</b>", h2_style))
    story.append(Image("output_plots/task1_distributions.png", width=520, height=195))
    story.append(Spacer(1, 3))
    obs_task1 = (
        "<b>Key Statistical Observations:</b> (1) The tilt angles feature 19 unique discrete values. 0&deg; is sampled 3 times, while intermediate angles (&plusmn;10&deg; to &plusmn;80&deg;) "
        "appear twice (forward ascending and reverse descending branches). (2) For axial tilt, &Delta;&lambda;<sub>1</sub> and &Delta;&lambda;<sub>2</sub> display strict push-pull "
        "anti-symmetry (&Delta;&lambda;<sub>1</sub> &asymp; -&Delta;&lambda;<sub>2</sub>), causing &Delta;&lambda;<sub>diff</sub> to span [-71 pm, +68 pm] while &Delta;&lambda;<sub>avg</sub> remains tightly centered at 0 pm (&sigma;=0.68 pm)."
    )
    story.append(Paragraph(obs_task1, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: DATA PREPROCESSING (TASK 2) & FEATURE ENGINEERING (TASK 3)
    # =========================================================================
    story.append(Paragraph("3. Task 2: Data Visualization and Preprocessing", h1_style))
    task2_text = (
        "Sensor response curves reveal the fundamental physical characteristics of the biaxial tilt mechanism. In axial tilt, bending places one FBG in tension "
        "and the opposing FBG in compression, generating monotonic anti-symmetric wavelength shifts. Conversely, in transverse tilt, both FBGs experience "
        "symmetric tensile strain, producing positive quadratic shifts regardless of tilt sign."
    )
    story.append(Paragraph(task2_text, body_style))
    
    # Scatter Responses & Boxplots side by side or stacked
    story.append(Paragraph("<b>Figure 2: FBG Peak Wavelength Responses Against Tilt Angle (Axial & Transverse)</b>", h2_style))
    story.append(Image("output_plots/task2_scatter_responses.png", width=520, height=155))
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("<b>Figure 3: Correlation Heatmaps & Outlier Detection Boxplots</b>", h2_style))
    story.append(Image("output_plots/task2_correlation_heatmaps.png", width=520, height=155))
    story.append(Spacer(1, 2))
    
    # Preprocessing & Normalization note
    preproc_note = (
        "<b>Data Preprocessing & Leakage Control:</b> Both <code>StandardScaler</code> (z-score standardization) and <code>MinMaxScaler</code> [0, 1] were evaluated. "
        "To prevent severe data leakage, <i>all scaling transformers are strictly fitted only on the training folds</i> and subsequently applied to test folds. "
        "Boxplot analysis confirms absence of anomalous outliers beyond sensor physical thresholds (all values reside within 1.5&times;IQR)."
    )
    story.append(Paragraph(preproc_note, body_style))
    story.append(Spacer(1, 3))
    
    story.append(Paragraph("4. Task 3: Feature Engineering and Feature Selection", h1_style))
    task3_text = (
        "Engineered features were derived from raw wavelength shifts: (1) Differential wavelength change <b>&Delta;&lambda;<sub>diff</sub> = &Delta;&lambda;<sub>1</sub> - &Delta;&lambda;<sub>2</sub></b>, "
        "and (2) Average wavelength change <b>&Delta;&lambda;<sub>avg</sub> = (&Delta;&lambda;<sub>1</sub> + &Delta;&lambda;<sub>2</sub>) / 2</b>. "
        "Four candidate feature subsets were evaluated: <b>F1</b>: [&Delta;&lambda;<sub>1</sub>, &Delta;&lambda;<sub>2</sub>], <b>F2</b>: [&Delta;&lambda;<sub>diff</sub>], "
        "<b>F3</b>: [&Delta;&lambda;<sub>avg</sub>], and <b>F4</b>: [&Delta;&lambda;<sub>diff</sub>, &Delta;&lambda;<sub>avg</sub>]."
    )
    story.append(Paragraph(task3_text, body_style))
    
    # Correlation table
    corr_table_data = [
        [Paragraph(h, table_header) for h in ["Feature Configuration", "Feature Variables", "Axial Corr (r with &theta;)", "Transverse Corr (r with &theta;)", "Transverse Corr (r with |&theta;|)"]],
        [Paragraph("F1: Individual Sensors", table_cell), Paragraph("&Delta;&lambda;<sub>1</sub>, &Delta;&lambda;<sub>2</sub>", table_cell), "+0.9917 / -0.9939", "+0.1082 / +0.1288", "+0.9725 / +0.9656"],
        [Paragraph("F2: Differential Change", table_cell), Paragraph("&Delta;&lambda;<sub>diff</sub>", table_cell), "+0.9932 (Rank 1)", "-0.2007", "+0.9010"],
        [Paragraph("F3: Average Change", table_cell), Paragraph("&Delta;&lambda;<sub>avg</sub>", table_cell), "+0.1135", "+0.1183", "+0.9693"],
        [Paragraph("F4: Joint Representation", table_cell), Paragraph("&Delta;&lambda;<sub>diff</sub>, &Delta;&lambda;<sub>avg</sub>", table_cell), "Dual High Corr", "Mixed Sign", "Dual High Magnitude"]
    ]
    t_corr = Table(corr_table_data, colWidths=[110, 110, 95, 95, 110])
    t_corr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_corr)
    story.append(Spacer(1, 2))
    task3_conclusion = (
        "<b>Feature Selection Decision:</b> For axial tilt, <b>F2 (&Delta;&lambda;<sub>diff</sub>)</b> and <b>F1 (&Delta;&lambda;<sub>1</sub>, &Delta;&lambda;<sub>2</sub>)</b> provide the highest linear correlation (|r| > 0.993) "
        "because common-mode thermal drift is eliminated. For transverse tilt, linear correlation with signed angle is negligible (|r| &le; 0.20), but correlation with absolute tilt |&theta;| reaches <b>r = 0.9725</b>, "
        "confirming that the transverse sensor behaves as a symmetric absolute inclination detector."
    )
    story.append(Paragraph(task3_conclusion, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: METHODOLOGY ARCHITECTURE & STATISTICAL REGRESSION (TASK 4)
    # =========================================================================
    story.append(Paragraph("5. Methodology and Pipeline Architecture", h1_style))
    story.append(Paragraph("<b>Figure 4: End-to-End Leakage-Free Grouped Cross-Validation Pipeline</b>", h2_style))
    story.append(Image("output_plots/methodology_diagram.png", width=520, height=135))
    story.append(Spacer(1, 3))
    
    story.append(Paragraph("6. Grouped Cross-Validation Algorithm Formulation", h1_style))
    algo_box = [
        [Paragraph("<b>Algorithm 1: Leak-Free Grouped 5-Fold Cross-Validation with In-Fold Model Optimization</b>", table_header)],
        [Paragraph(
            "<b>Input:</b> Dataset <i>D</i> = {(<b>x</b><sub>i</sub>, y<sub>i</sub>, g<sub>i</sub>)}<sub>i=1</sub><sup>N</sup> where g<sub>i</sub> = tilt angle group; Candidate models <i>M</i>; Parameter grids &Omega;.<br/>"
            "<b>Output:</b> Out-of-Fold predictions <i>y&#770;</i>, evaluation metrics (MAE, MSE, RMSE, R&sup2;), training latency <i>T</i>.<br/>"
            "1: Partition distinct groups <i>G</i> = unique(g) into 5 disjoint subsets {G<sub>k</sub>}<sub>k=1</sub><sup>5</sup> using GroupKFold.<br/>"
            "2: <b>For</b> each fold <i>k</i> &in; {1, ..., 5}:<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;a: Partition train set <i>D</i><sub>train</sub> = {i | g<sub>i</sub> &notin; G<sub>k</sub>} and test set <i>D</i><sub>test</sub> = {i | g<sub>i</sub> &in; G<sub>k</sub>}.<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;b: Fit <code>StandardScaler</code> &mu;<sub>train</sub>, &sigma;<sub>train</sub> solely on <i>X</i><sub>train</sub>; transform <i>X&#771;</i><sub>train</sub> and <i>X&#771;</i><sub>test</sub>.<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;c: <b>In-Fold Hyperparameter Tuning:</b> Conduct 3-fold inner GroupKFold on <i>D</i><sub>train</sub> to select &theta;* = argmin<sub>&theta;&in;&Omega;</sub> MSE<sub>inner</sub>.<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;d: Train estimator <i>m</i>(X&#771;; &theta;*) on full <i>D</i><sub>train</sub>; generate predictions <i>y&#770;</i><sub>test</sub> = <i>m</i>(<i>X&#771;</i><sub>test</sub>).<br/>"
            "3: Aggregate all out-of-fold predictions; compute global metrics MAE = (1/N)&sum;|y<sub>i</sub> - y&#770;<sub>i</sub>|, RMSE = &radic;((1/N)&sum;(y<sub>i</sub> - y&#770;<sub>i</sub>)&sup2;), R&sup2; = 1 - (&sum;(y<sub>i</sub> - y&#770;<sub>i</sub>)&sup2; / &sum;(y<sub>i</sub> - y&#772;)&sup2;).",
            table_cell
        )]
    ]
    t_algo = Table(algo_box, colWidths=[520])
    t_algo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d47a1")),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f8f9fa")),
        ('GRID', (0,0), (-1,-1), 0.8, colors.HexColor("#90caf9")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_algo)
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("7. Task 4: Traditional Statistical Regression Models", h1_style))
    task4_text = (
        "Six traditional statistical regression models were implemented and analyzed: "
        "<b>1. Simple Linear Regression (SLR):</b> y = &beta;<sub>0</sub> + &beta;<sub>1</sub>&Delta;&lambda;<sub>diff</sub>; "
        "<b>2. Multiple Linear Regression (MLR):</b> y = &beta;<sub>0</sub> + &beta;<sub>1</sub>&Delta;&lambda;<sub>1</sub> + &beta;<sub>2</sub>&Delta;&lambda;<sub>2</sub>; "
        "<b>3. Polynomial Regression:</b> Degrees 2 and 3 feature expansion; "
        "<b>4. Ridge Regression (L2):</b> Loss = ||y - X&beta;||&sup2; + &alpha;||&beta;||&sup2;; "
        "<b>5. Lasso Regression (L1):</b> Loss = ||y - X&beta;||&sup2; + &alpha;||&beta;||<sub>1</sub>; "
        "<b>6. Elastic Net:</b> Combined penalty &alpha;&rho;||&beta;||<sub>1</sub> + 0.5&alpha;(1-&rho;)||&beta;||&sup2;. "
        "Regularization penalties (&alpha; &in; [0.01, 10.0]) and degrees were tuned strictly in-fold."
    )
    story.append(Paragraph(task4_text, body_style))
    story.append(Spacer(1, 2))
    
    # Statistical Actual vs Predicted plot
    story.append(Paragraph("<b>Figure 5: Actual vs Predicted Angles: Best Statistical Model (Polynomial Degree 3)</b>", h2_style))
    story.append(Image("output_plots/graph4_actual_vs_predicted_stat.png", width=340, height=160))
    story.append(Spacer(1, 2))
    
    stat_findings = (
        "<b>Statistical Regression Analysis:</b> Polynomial Regression (degree 3) achieved exceptional performance with <b>RMSE = 5.139&deg;</b> and <b>R&sup2; = 0.9900</b>, "
        "capturing subtle mechanical non-linearities near extreme angles (&plusmn;80&deg;, &plusmn;90&deg;). "
        "Lasso and Elastic Net effectively matched standard Linear Regression (RMSE 6.48&deg; - 6.58&deg;) with mild shrinkage, preventing overfitting."
    )
    story.append(Paragraph(stat_findings, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: CONVENTIONAL MACHINE LEARNING MODELS (TASK 5)
    # =========================================================================
    story.append(Paragraph("8. Task 5: Conventional Machine Learning Regression Models", h1_style))
    task5_text = (
        "Six non-parametric and ensemble learning algorithms were evaluated across specified hyperparameter grids: "
        "(1) <b>Support Vector Regression (SVR):</b> Linear and RBF kernels; "
        "(2) <b>Decision Tree Regression (DTR):</b> Max depth &in; {2, 3, 5}; "
        "(3) <b>Random Forest Regression (RFR):</b> Trees &in; {50, 100}; "
        "(4) <b>K-Nearest Neighbors (KNN):</b> Neighbors <i>k</i> &in; {3, 5, 7}; "
        "(5) <b>Gradient Boosting Regression (GBR):</b> Learning rate &in; {0.01, 0.1}; "
        "(6) <b>Gaussian Process Regression (GPR):</b> RBF and Mat&eacute;rn kernels."
    )
    story.append(Paragraph(task5_text, body_style))
    story.append(Spacer(1, 3))
    
    # Hyperparameter comparison table
    story.append(Paragraph("<b>Table 3: Hyperparameter Search Space, In-Fold Optimal Configurations, and Model Behavior</b>", h2_style))
    hyper_table_data = [
        [Paragraph(h, table_header) for h in ["Model", "Hyperparameters Explored", "Optimal In-Fold Selection", "Axial RMSE (&deg;)", "Axial R&sup2;", "Hyperparameter Sensitivity & Insights"]],
        [Paragraph("SVR", table_cell), Paragraph("Kernel: Linear vs RBF; C=1.0", table_cell), "Linear Kernel", "10.293", "0.9599", Paragraph("Linear kernel outperformed RBF due to strong global linearity of axial strain.", table_cell)],
        [Paragraph("Decision Tree", table_cell), Paragraph("max_depth &in; {2, 3, 5}", table_cell), "max_depth = 5", "10.000", "0.9622", Paragraph("Depth 2 underfits badly; depth 5 captures step functions but suffers from discretization.", table_cell)],
        [Paragraph("Random Forest", table_cell), Paragraph("n_estimators &in; {50, 100}", table_cell), "n_estimators = 100", "6.935", "0.9818", Paragraph("Ensemble averaging smoothed tree discontinuities, cutting RMSE by 31% over single tree.", table_cell)],
        [Paragraph("KNN Regressor", table_cell), Paragraph("n_neighbors &in; {3, 5, 7}", table_cell), "n_neighbors = 3", "6.102", "0.9859", Paragraph("Small <i>k</i>=3 preserves local angular gradients without over-smoothing neighboring angles.", table_cell)],
        [Paragraph("Gradient Boosting", table_cell), Paragraph("learning_rate &in; {0.01, 0.1}", table_cell), "learning_rate = 0.1", "8.604", "0.9720", Paragraph("lr=0.01 suffered from underfitting given 100 estimators; lr=0.1 converged effectively.", table_cell)],
        [Paragraph("Gaussian Process", table_cell), Paragraph("Kernel: RBF vs Mat&eacute;rn", table_cell), "Mat&eacute;rn Kernel", "5.032", "0.9904", Paragraph("Mat&eacute;rn kernel provides ideal flexibility, achieving the lowest overall ML RMSE (5.032&deg;).", table_cell)]
    ]
    t_hyper = Table(hyper_table_data, colWidths=[65, 95, 80, 52, 45, 183])
    t_hyper.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
        ('ALIGN', (3,1), (4,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_hyper)
    story.append(Spacer(1, 4))
    
    # Best ML Model actual vs predicted and residuals
    story.append(Paragraph("<b>Figure 6: Best Conventional ML Model (GPR) Actual vs Predicted & Dual Residual Analysis</b>", h2_style))
    story.append(Image("output_plots/graph6_residual_plots.png", width=520, height=170))
    story.append(Spacer(1, 3))
    
    ml_eval_text = (
        "<b>Residual and Error Analysis:</b> Residual distributions for both Polynomial Regression and Gaussian Process Regression (GPR) are centered around "
        "zero residual error with standard deviation &sigma; &asymp; 5&deg;. Maximum localized errors occur at the transition points (&plusmn;90&deg;) where the cantilever beam "
        "approaches mechanical inflection. GPR with the Mat&eacute;rn kernel demonstrates superior generalization due to its continuous probabilistic interpolation."
    )
    story.append(Paragraph(ml_eval_text, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: PERFORMANCE EVALUATION & COMPARATIVE ANALYSIS (TASK 6)
    # =========================================================================
    story.append(Paragraph("9. Task 6: Performance Evaluation and Comparative Analysis", h1_style))
    task6_desc = (
        "Rigorous 5-fold grouped cross-validation results across all eleven models are summarized below. "
        "Metrics reported are Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), "
        "Coefficient of Determination (R&sup2;), and cumulative training latency (s)."
    )
    story.append(Paragraph(task6_desc, body_style))
    
    # Axial Results Table
    story.append(Paragraph("<b>Table 4: Comparative Regression Performance - Axial Tilt Angle Prediction</b>", h2_style))
    df_ax = pd.read_csv('output_plots/axial_results.csv', index_col=0)
    ax_table_data = [
        [Paragraph(h, table_header) for h in ["Regression Model", "MAE (&deg;)", "MSE (&deg;&sup2;)", "RMSE (&deg;)", "R&sup2; Score", "Training Time (s)", "Selected Hyperparameter"]]
    ]
    for idx, row in df_ax.iterrows():
        ax_table_data.append([
            Paragraph(idx, table_cell),
            f"{row['MAE']:.3f}",
            f"{row['MSE']:.2f}",
            f"{row['RMSE']:.3f}",
            f"{row['R2']:.4f}",
            f"{row['Training Time (s)']:.4f}",
            Paragraph(str(row['Hyperparams']) if str(row['Hyperparams']) != 'nan' else 'Standard', table_cell)
        ])
    t_ax = Table(ax_table_data, colWidths=[105, 52, 55, 55, 58, 75, 120])
    t_ax.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0d47a1")),
        ('ALIGN', (1,1), (5,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_ax)
    story.append(Spacer(1, 3))
    
    # Transverse Absolute Results Table
    story.append(Paragraph("<b>Table 5: Comparative Regression Performance - Transverse Tilt Magnitude Prediction (|&theta;|)</b>", h2_style))
    df_tr_abs = pd.read_csv('output_plots/transverse_abs_results.csv', index_col=0)
    tr_table_data = [
        [Paragraph(h, table_header) for h in ["Regression Model", "MAE (&deg;)", "MSE (&deg;&sup2;)", "RMSE (&deg;)", "R&sup2; Score", "Training Time (s)", "Selected Hyperparameter"]]
    ]
    for idx, row in df_tr_abs.iterrows():
        tr_table_data.append([
            Paragraph(idx, table_cell),
            f"{row['MAE']:.3f}",
            f"{row['MSE']:.2f}",
            f"{row['RMSE']:.3f}",
            f"{row['R2']:.4f}",
            f"{row['Training Time (s)']:.4f}",
            Paragraph(str(row['Hyperparams']) if str(row['Hyperparams']) != 'nan' else 'Standard', table_cell)
        ])
    t_tr = Table(tr_table_data, colWidths=[105, 52, 55, 55, 58, 75, 120])
    t_tr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2e7d32")),
        ('ALIGN', (1,1), (5,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f5f5")]),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ]))
    story.append(t_tr)
    story.append(Spacer(1, 3))
    
    # Comparative Graphs (Graphs 1, 2, 7)
    story.append(Paragraph("<b>Figure 7: Model-wise Performance Metrics Comparison (MAE, RMSE, Training Latency)</b>", h2_style))
    story.append(Image("output_plots/graph2_rmse_comparison.png", width=255, height=130))
    # Place training time right next to it or stacked
    story.append(Image("output_plots/graph7_training_time_comparison.png", width=520, height=125))
    story.append(Spacer(1, 2))
    
    story.append(Paragraph(
        "<b>Comparative Summary:</b> (1) <b>GPR (Mat&eacute;rn)</b> achieved the lowest overall RMSE (5.032&deg;) and highest R&sup2; (0.9904) for axial tilt, "
        "closely followed by <b>Polynomial Regression (Degree 3)</b> (RMSE 5.139&deg;, R&sup2; 0.9900). "
        "(2) <b>Linear/Lasso Regression</b> offers an ultra-fast training time (0.012s) with near-optimal accuracy (RMSE 6.58&deg;), "
        "making it 50&times; faster than Random Forest (0.780s) and ideal for embedded edge execution.",
        body_style
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: ADVANCED TASKS, QUESTION RESPONSES (TASK 7), CODE LINK & CONCLUSION
    # =========================================================================
    story.append(Paragraph("10. Advanced Laboratory Tasks", h1_style))
    
    # Advanced Tasks A, B, C summaries
    adv_box = [
        [Paragraph("<b>Advanced Task A: Noise Robustness</b>", table_header),
         Paragraph("<b>Advanced Task B: Sensor Hysteresis</b>", table_header),
         Paragraph("<b>Advanced Task C: Signed vs Absolute Tilt</b>", table_header)],
        [
            Paragraph("Gaussian noise (&sigma; = 0.5 to 5.0 pm) injected strictly into evaluation inputs. Ridge and GPR demonstrated superior noise filtering; Linear Regression degraded gracefully from 6.68&deg; to 8.42&deg;, while Random Forest suffered higher variance.", table_cell),
            Paragraph("Ascending (0&deg;&rarr;90&deg;) vs descending (90&deg;&rarr;0&deg;) cycles revealed maximum hysteresis &Delta;&lambda;<sub>hyst</sub> = 2.0 pm at 0&deg; and 20&deg;, corresponding to a slight angular lag of ~1.2&deg; due to mechanical cantilever relaxation.", table_cell),
            Paragraph("FBG sensor physics induces symmetric tensile strain in transverse bending. Predicting signed &theta; yielded R&sup2; &le; 0.00; whereas predicting |&theta;| yielded <b>R&sup2; = 0.9662</b> and <b>RMSE = 4.957&deg;</b>, confirming pure inclination magnitude sensing.", table_cell)
        ]
    ]
    t_adv = Table(adv_box, colWidths=[173, 173, 174])
    t_adv.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1a237e")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#b0bec5")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f8f9fa")]),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_adv)
    story.append(Spacer(1, 3))
    
    # Advanced tasks plots
    story.append(Paragraph("<b>Figure 8: Advanced Task A (Noise Robustness) & Task B (Hysteresis Loop)</b>", h2_style))
    # side by side images
    img_data = [
        [Image("output_plots/advanced_taskA_noise_robustness.png", width=255, height=125),
         Image("output_plots/advanced_taskB_hysteresis.png", width=255, height=125)]
    ]
    t_imgs = Table(img_data, colWidths=[260, 260])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 2))
    
    story.append(Paragraph("11. Task 7: Comprehensive Answers to Evaluator Questions", h1_style))
    q_answers = [
        "<b>Q1. Best Traditional Statistical Model:</b> <b>Polynomial Regression (Degree 3)</b> achieved the best accuracy with <b>MAE = 3.454&deg;</b>, <b>RMSE = 5.139&deg;</b>, and <b>R&sup2; = 0.9900</b>, effectively fitting slight curvature at high tilt.",
        "<b>Q2. Conventional ML Model with Lowest RMSE:</b> <b>Gaussian Process Regression (GPR with Mat&eacute;rn kernel)</b> attained the lowest RMSE of <b>5.032&deg;</b> (MAE = 3.664&deg;, R&sup2; = 0.9904), followed closely by <b>KNN (k=3)</b> with RMSE = 6.102&deg;.",
        "<b>Q3. Nonlinear vs Linear Algorithms:</b> Yes, nonlinear models (Degree 3 Poly Reg and GPR) outperformed linear models (RMSE 5.03&deg; - 5.14&deg; vs 6.68&deg;), capturing non-linear mechanical deformation.",
        "<b>Q4. Influence of Feature Selection:</b> Highly decisive. The differential feature &Delta;&lambda;<sub>diff</sub> yields r = 0.9932 and cancels common-mode temperature fluctuations, whereas raw individual wavelengths are more susceptible to offset errors.",
        "<b>Q5. Model Complexity vs Accuracy:</b> No. Highly complex models like Decision Trees (depth 5, RMSE 10.00&deg;) and Gradient Boosting (RMSE 8.60&deg;) exhibited overfitting and step-like errors on the 37-point dataset, underperforming regularized linear and polynomial models.",
        "<b>Q6. Difference Between Axial and Transverse Errors:</b> In axial tilt, the two FBGs undergo differential tension-compression (anti-symmetric, monotonic). In transverse tilt, bending geometry stretches both FBGs identically in both directions, making signed tilt physically ambiguous without absolute magnitude transformation.",
        "<b>Q7. Real-Time Deployment Recommendation:</b> <b>Ridge Regression (&alpha;=0.01) or Linear Regression on &Delta;&lambda;<sub>diff</sub></b> is recommended. It delivers near-optimal accuracy (RMSE 6.68&deg;, R&sup2; 0.9831) while requiring only two scalar operations (&Oscr;(1) inference, latency < 0.01 ms), ensuring ultra-reliable real-time micro-controller execution."
    ]
    for qa in q_answers:
        story.append(Paragraph(qa, body_style))
    story.append(Spacer(1, 3))
    
    # Public code link and submission info
    story.append(Paragraph("12. Public Source Code Repository & Verification Link", h1_style))
    code_text = (
        "All experimental pipelines, grouped cross-validation routines, and report scripts are publicly available at:<br/>"
        "&bull; <b>Public GitHub Repository:</b> <font color='#0d47a1'><u>https://github.com/MohithChandra07/ML-Lab-CS3103</u></font><br/>"
        "&bull; <b>Direct Submission Form Link:</b> <font color='#0d47a1'><u>https://forms.gle/RCiyByKQtEbeoNVB7</u></font><br/>"
        "&bull; <b>Reproducibility:</b> Generated via Python 3.14 with Scikit-Learn 1.9, Pandas 3.0, and ReportLab 5.0 on macOS arm64."
    )
    story.append(Paragraph(code_text, body_style))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    
    # Verify exact page count
    reader = pypdf.PdfReader(output_filename)
    page_count = len(reader.pages)
    print(f"Report generated successfully: '{output_filename}' with EXACTLY {page_count} pages.")
    return page_count

if __name__ == '__main__':
    cnt = create_report()
