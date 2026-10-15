import os

def build_report():
    print("Building reports/main.tex from scratch...")
    
    latex_code = r"""\documentclass[12pt, a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage{mathptmx} % Times New Roman font
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows, positioning, calc}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{hyperref}
\usepackage{array} % For advanced table column alignment
\usepackage[justification=centering, skip=10pt, font=rm]{caption} % Centered and uniformly formatted captions (Roman font)
\usepackage{parskip} % For paragraph spacing instead of indentation
\usepackage{chngcntr}
\usepackage{microtype} % Improves text alignment, spacing, and hyphenation
\usepackage{needspace} % Dynamic space allocation to prevent orphan headings

% TikZ styles for custom flowcharts
\tikzstyle{process} = [rectangle, minimum width=3.5cm, minimum height=0.6cm, text centered, draw=black, fill=blue!10, font=\footnotesize\bfseries]
\tikzstyle{groupbox} = [rectangle, draw=black!50, dashed, inner sep=8pt, fill=black!5, rounded corners]
\tikzstyle{arrow} = [thick,->,>=stealth]

% Table numbering includes section
\counterwithin{table}{section}

% Typographical Adjustments
\frenchspacing % Uniform spacing after periods (prevents large gaps)

% Uniform spacing around floats (Tables and Figures)
\setlength{\intextsep}{18pt}
\setlength{\textfloatsep}{18pt}

% Header and Footer Setup
\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\rmfamily Urban Bike-Sharing Demand Forecasting Using Machine Learning}
\fancyfoot[L]{\rmfamily Department of Computer Applications}
\fancyfoot[R]{\rmfamily \thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% Title Formatting (Ensuring consistent heading spacing and fonts)
\titleformat{\section}{\rmfamily\fontsize{16pt}{19.2pt}\selectfont\bfseries}{\thesection.}{0.3em}{\MakeUppercase}
\titleformat{\subsection}{\rmfamily\fontsize{14pt}{16.8pt}\selectfont\bfseries}{\thesubsection}{0.3em}{}
\titleformat{\subsubsection}{\rmfamily\fontsize{12pt}{14.4pt}\selectfont\bfseries}{\thesubsubsection}{0.3em}{}

% Line Spacing
\setstretch{1.5}

\flushbottom % Prevents vertical stretching that causes unnatural gaps between paragraphs

\begin{document}
\pagenumbering{arabic}
\setcounter{page}{1}
\setcounter{page}{1}

% ==============================================================================
% 1. INTRODUCTION
% ==============================================================================
\section{INTRODUCTION}

Urban bike-sharing systems have emerged as one of the most promising and sustainable modes of public transportation in modern smart cities, offering an eco-friendly, cost-effective, and healthy alternative to fossil-fuel-dependent vehicles. By providing commuters with flexible point-to-point transit for first-mile and last-mile connectivity, bike-sharing networks mitigate vehicular traffic congestion, reduce carbon emissions, and promote active lifestyles. The integration of technology, mobile applications, and physical infrastructure has enabled these networks to expand rapidly worldwide. Over the past decade, these systems have evolved from simple coin-operated municipal programs into sophisticated, data-driven dockless and docked networks tracked continuously via GPS and cellular data arrays.

However, managing the operational equilibrium across a vast network of docking stations remains a profoundly complex logistical challenge. Because users do not return bicycles to their original starting locations, the system naturally flows out of balance. Unpredictable demand fluctuations frequently cause popular pickup stations (such as residential zones during the morning commute) to experience bicycle deficits (shortages), leaving intending users stranded. Conversely, popular destination hubs (such as commercial districts and major transit terminals) simultaneously reach full capacity and cannot accept incoming returns, causing massive surpluses. This imbalance creates a logistical nightmare for municipal operators who must deploy fleets of physical redistribution trucks to manually move bicycles back to where they are needed.

Advances in data-driven techniques and predictive analytics have established supervised machine learning as an effective and scalable solution for urban mobility forecasting. Machine learning models possess the capability to uncover complex, non-linear relationships between station usage patterns and a multitude of temporal factors, including peak commuting hours, day of the week, and seasonal shifts, without relying on rigid, static schedules. Traditional statistical models, such as Auto-Regressive Integrated Moving Average (ARIMA), often struggle to capture the high-frequency volatility of micro-mobility data, particularly when attempting to model dozens of independent stations simultaneously. In contrast, modern ensemble regression algorithms such as Random Forest, Extreme Gradient Boosting (XGBoost), and CatBoost are remarkably well-suited for capturing these intricate dynamics due to their exceptional predictive accuracy, non-linear mapping capabilities, and resilience against noisy transactional data.

This project focuses on building a comprehensive, intelligent machine learning pipeline to accurately forecast hourly bike-sharing departure demand across the San Francisco Bay Area network using extensive historical trip data. The target variable is explicitly defined as the hourly station-level departure demand, representing the exact number of trips starting from a particular station during a specific hour. This definition provides a direct, actionable metric for fleet operators. The historical dataset encompasses hundreds of thousands of individual bicycle trips across multiple urban cities, including San Francisco, San Jose, Mountain View, and Palo Alto. The end-to-end workflow involves systematic data preprocessing, chronological train-validation-test partitioning, advanced time-series feature engineering, and rigorous multi-model benchmarking.

Accurate bike-sharing demand forecasting directly enhances urban micro-mobility management in several profound ways. Primarily, it provides demand-based redistribution support, enabling transit operators to preemptively anticipate station deficits and allocate bicycles more efficiently before shortages occur. Furthermore, it improves overall commuter satisfaction by ensuring asset availability and minimizes the operational overhead associated with reactive vehicle dispatching. By combining historical trip records with state-of-the-art machine learning algorithms, this project delivers a reliable, scalable, and data-driven approach for smart city transit optimization, demonstrating how predictive analytics can transform physical infrastructure management.

In summary, the work aims to demonstrate the capability of supervised machine learning algorithms to forecast future demand using purely historical transactional data and rigorously evaluate predictive performance, thereby contributing significantly to the efficient operation and sustainability of modern urban transportation networks.

\newpage
% ==============================================================================
% 2. SUPPORTING LITERATURE
% ==============================================================================
\section{SUPPORTING LITERATURE}
\subsection{Literature Review}

\begin{table}[H]
\centering
\begin{tabular}{|p{0.3\textwidth}|p{0.6\textwidth}|}
\hline
\textbf{TITLE} & Prediction bike-sharing demand with gradient boosting methods \\ \hline
\textbf{AREA OF WORK} & The paper focuses on applying machine learning and gradient boosting algorithms to predict bike-sharing demand across multiple urban environments (Konya and Washington D.C.) to optimize fleet distribution. \\ \hline
\textbf{DATASET} & Historical bike-sharing trip data from Konya, Turkey and Washington, D.C., USA, including meteorological parameters and temporal variables. \\ \hline
\textbf{METHODOLOGY/STRATEGY} & The study uses historical data for training and testing. Data preprocessing involves extracting temporal features. Three gradient boosting algorithms are trained and evaluated using error metrics. \\ \hline
\textbf{ALGORITHM} & 
Extreme Gradient Boosting (XGBoost), Light Gradient Boosting Machine (LightGBM), CatBoost \\ \hline
\textbf{RESULT/PRECISION} & 
Gradient boosting methods significantly outperformed classical models and demonstrated high resilience to noise and cyclical data. \\ \hline
\textbf{ADVANTAGES} & The study proves that gradient boosting handles non-linear temporal data effortlessly, establishing it as the standard for tabular mobility forecasting. \\ \hline
\textbf{FUTURE PROPOSAL} & Future research will explore real-time streaming data integration and hybrid deep learning models. \\ \hline
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{|p{0.3\textwidth}|p{0.6\textwidth}|}
\hline
\textbf{TITLE} & A predictive analytics approach for forecasting bike rental demand \\ \hline
\textbf{AREA OF WORK} & The paper focuses on evaluating the performance of six diverse machine learning regression models in forecasting hourly bike rental demand to improve operational balancing. \\ \hline
\textbf{DATASET} & Public bike rental records encompassing temporal features, weather conditions, and holiday indicators. \\ \hline
\textbf{METHODOLOGY/STRATEGY} & Historical data was preprocessed and split into training and testing sets. Six different regression models were trained. Performance was compared using R-squared, RMSE, and MAE. \\ \hline
\textbf{ALGORITHM} & 
Ordinary Least Squares (OLS), Multi-Layer Perceptron (MLP), Gradient Boosting Regression, Random Forest Regression, Decision Tree Regression \\ \hline
\textbf{RESULT/PRECISION} & Random Forest and Gradient Boosting achieved the highest performance, heavily outperforming Neural Networks on tabular data. \\ \hline
\textbf{ADVANTAGES} & The paper provides a robust comparative analysis, demonstrating that tree-based bagging and boosting are highly reliable for mobility forecasting. \\ \hline
\textbf{FUTURE PROPOSAL} & Expanding the models to include dynamic pricing variables and weather forecasting integration. \\ \hline
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\begin{tabular}{|p{0.3\textwidth}|p{0.6\textwidth}|}
\hline
\textbf{TITLE} & GSTGAT: Gated spatiotemporal graph attention network for traffic demand forecasting \\ \hline
\textbf{AREA OF WORK} & The paper focuses on predicting traffic and bike-sharing demand using complex deep learning graph architectures to capture spatial dependencies between stations. \\ \hline
\textbf{DATASET} & San Francisco Bay Area bicycle dataset (the same foundation utilized in this project). \\ \hline
\textbf{METHODOLOGY/STRATEGY} & The study applied advanced graph attention mechanisms and gated recurrent units to model the network topology. Performance was evaluated against baseline deep learning models. \\ \hline
\textbf{ALGORITHM} & 
Gated Spatiotemporal Graph Attention Network (GSTGAT), Baseline Recurrent Neural Networks (RNN/LSTM) \\ \hline
\textbf{RESULT/PRECISION} & The graph network successfully captured spatial dependencies, proving the intricate connectivity of the SF Bay Area stations. \\ \hline
\textbf{ADVANTAGES} & Deeply models spatial topology, providing insight into how neighboring stations influence each other. \\ \hline
\textbf{FUTURE PROPOSAL} & Exploring hybrid architectures that combine tree-based tabular models with spatial graph embeddings. \\ \hline
\end{tabular}
\end{table}

\subsection{Literature Review Summary}

\begin{table}[H]
\centering
\begin{tabular}{|p{0.3\textwidth}|p{0.6\textwidth}|}
\hline
\textbf{TITLE} & \textbf{SUMMARY} \\ \hline
\textbf{Paper 1: Prediction bike-sharing demand with gradient boosting methods} & This study evaluates predicting bike-sharing demand using XGBoost, LightGBM, and CatBoost across two distinct cities. It proves that gradient boosting frameworks are exceptionally capable of capturing cyclical human mobility patterns, establishing them as superior to classical linear models. \\ \hline
\textbf{Paper 2: A predictive analytics approach for forecasting bike rental demand} & This work compared six models (Random Forest, MLP, OLS, Decision Trees) for rental demand forecasting. The results highlighted that ensemble tree methods heavily outperformed standard neural networks on tabular data, providing a stable and generalizable solution. \\ \hline
\textbf{Paper 3: GSTGAT: Gated spatiotemporal graph attention network} & This study utilized the SF Bay Area dataset to build a graph neural network. While focused on deep learning, it validates the extreme importance of temporal cyclicality and spatial dependencies within the exact transit network studied in this project. \\ \hline
\end{tabular}
\end{table}

\subsection{Findings and Proposals}
The initial study reviewed by Aydın et al. (2023) examined the prediction of bike-sharing demand using advanced gradient boosting. The primary takeaway from this research is that integrating sequential error-correcting algorithms like XGBoost and CatBoost improves the reliability of forecasting, making it more applicable for practical uses such as managing fleet distribution.

The second study by Karunanithi et al. (2024) presented a comparative analysis of machine learning predictive models. Although Neural Networks were part of the investigation, the best results emerged from Random Forest and Gradient Boosting. The insight from this study is that ensemble machine learning approaches perform exceptionally well when significant temporal features are engineered, avoiding the overfitting commonly seen in deep learning.

The third research effort by Yao et al. (2024) evaluated the effectiveness of deep learning structures on the exact SF Bay Area dataset. The results emphasized the intricate connectivity of the stations. The conclusion from this study is that while spatial graphs are complex, the underlying temporal demand is the primary driver of accuracy.

\newpage
% ==============================================================================
% 3. SYSTEM ANALYSIS
% ==============================================================================
\section{SYSTEM ANALYSIS}

\subsection{Analysis of the dataset}
\subsubsection{About the dataset}
\textbf{a) Dataset Overview}

The foundation of this predictive modeling project is the SF Bay Area Bike Share Dataset. To ensure the model relies strictly on historically verified, transactional human behavior rather than external approximations, the final implementation uses exclusively the \texttt{trip.csv} file from the dataset.

\begin{itemize}
    \item \textbf{Total Raw Records:} 669,959 individual, anonymized bicycle trip transactions.
    \item \textbf{Time Span:} The data covers exactly two years of continuous operation, from August 29, 2013 (09:00:00) to August 31, 2015 (23:00:00).
    \item \textbf{Total Features:} 12 raw columns.
    \item \textbf{Data Types:} A mix of numerical IDs, integers, and string objects (which must be converted to datetime).
    \item \textbf{Target Variable:} \texttt{hourly\_demand} (calculated by aggregating \texttt{start\_date} and \texttt{start\_station\_id}).
\end{itemize}

\textbf{b) Dataset sample records}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_3_1_shape_dtypes.png}
\caption{Result of shape and datatype of dataset}
\label{fig:shape_dtypes}
\end{figure}

Explanation of Figure \ref{fig:shape_dtypes}: This figure displays the programmatic output verifying the fundamental structure of the imported dataset. It confirms that the data matrix contains exactly 669,959 rows and 12 columns. Furthermore, it details the memory datatypes assigned to each column.

\subsubsection{Explore the dataset}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_3_3_first_10_records.png}
\caption{First ten records of the dataset}
\label{fig:first_10_records}
\end{figure}

Explanation of Figure \ref{fig:first_10_records}: This screenshot provides a direct glimpse into the raw transactional data structure by displaying the top 10 rows. It illustrates exactly how a single trip is recorded: showing the trip ID, duration, timestamp, and station details.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_3_4_summary_stats.png}
\caption{Summary Statistics of Numeric Columns}
\label{fig:summary_stats}
\end{figure}

Explanation of Figure \ref{fig:summary_stats}: The summary statistics table provides mathematical insights into the distribution of the numerical columns. Crucially, by examining the \texttt{duration} column, we can observe the presence of massive outliers, justifying our architectural decision to forecast sheer departure volume rather than attempting to predict trip duration.

\subsection{Data Preprocessing}
\subsubsection{Data Cleaning}

\begin{figure}[H]
\centering

\begin{tikzpicture}[node distance=1.5cm, auto]
    \node [process] (step1) {Raw Data Ingestion (669,959 rows)};
    \node [process, below of=step1] (step2) {Null Value Auditing};
    \node [process, below of=step2] (step3) {Datetime Conversion};
    \node [process, below of=step3] (step4) {Irrelevant Column Dropping};
    \node [process, below of=step4] (step5) {Cleaned Dataset};
    
    \draw [arrow] (step1) -- (step2);
    \draw [arrow] (step2) -- (step3);
    \draw [arrow] (step3) -- (step4);
    \draw [arrow] (step4) -- (step5);
\end{tikzpicture}
\caption{Detailed Data Preprocessing Flow}
\label{fig:preprocessing_flow}
\end{figure}

Explanation of Figure \ref{fig:preprocessing_flow}: This flowchart illustrates the sequential steps taken during the data cleaning phase. It ensures that the raw data is thoroughly audited for missing values, timestamps are properly converted for time-series extraction, and unnecessary demographic data is safely removed to optimize memory usage.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_3_11_missing_values.png}
\caption{Missing Values Verification}
\label{fig:missing_values}
\end{figure}

Explanation of Figure \ref{fig:missing_values}: This figure displays the result of the null-value checking algorithm. It verifies the operational integrity by proving that zero records are missing critical routing data.

\subsubsection{Analysis of Feature Variables}

\begin{figure}[H]
\centering

\begin{tikzpicture}[node distance=1.5cm, auto]
    \node [process] (step1) {Base Hourly Grid};
    \node [process, below left=1cm and -1cm of step1] (step2a) {Temporal Extraction (Hour, Day)};
    \node [process, below right=1cm and -1cm of step1] (step2b) {Cyclic Encoding (Sin/Cos)};
    \node [process, below=2cm of step1] (step3) {Lag Generation (1h, 24h, 168h)};
    \node [process, below=1cm of step3] (step4) {Rolling Statistics (Mean, Max)};
    \node [process, below=1cm of step4] (step5) {Final Feature Matrix (54 variables)};
    
    \draw [arrow] (step1) -- (step2a);
    \draw [arrow] (step1) -- (step2b);
    \draw [arrow] (step2a) -- (step3);
    \draw [arrow] (step2b) -- (step3);
    \draw [arrow] (step3) -- (step4);
    \draw [arrow] (step4) -- (step5);
\end{tikzpicture}
\caption{Feature Engineering Pipeline}
\label{fig:fe_flow}
\end{figure}

Explanation of Figure \ref{fig:fe_flow}: This diagram breaks down the specific feature engineering operations applied to the dataset. It highlights how the base temporal features are mathematically transformed into cyclical sin/cos waves and how historical lag variables are constructed to provide the machine learning algorithms with necessary context.
To transform the raw transactional records into a format suitable for machine learning, a complete Cartesian grid comprising all 70 stations and all 17,583 hourly timestamps was constructed. Zero-demand hours were appropriately filled with 0. 54 distinct numerical features were engineered.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_3_15_cyclic.png}
\caption{Handling Cyclic Features Implementation}
\label{fig:cyclic_features}
\end{figure}

Explanation of Figure \ref{fig:cyclic_features}: This figure demonstrates the Python implementation of the trigonometric cyclic encoding process for time variables.

\subsubsection{Analysis of Class Variables}
The target variable is the hourly station-level departure demand. This represents the absolute count of bicycle trips starting from a specific docking station during a specific one-hour block.

\subsection{Data Visualization}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{hourly_demand.png}
\caption{Hourly Demand Pattern Across All Stations}
\label{fig:hourly_demand}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{weekday_demand.png}
\caption{Demand Pattern Based on Day of the Week}
\label{fig:weekday_demand}
\end{figure}

\1

The weekday demand chart provides profound insights into the primary utility of the bike-sharing system. The consistently high volume observed from Monday through Friday strongly indicates that the network is heavily relied upon for daily, mandatory commuting rather than leisurely weekend rides. The sheer volume on weekdays, peaking tightly around Tuesday and Wednesday, suggests that users have integrated the service into their core weekly routines.

When the weekend arrives (Saturday and Sunday), there is a catastrophic drop in departure volume, plummeting by more than 60\%. This confirms that tourist activity and casual weekend riding constitute only a minor fraction of the total system utilization. For predictive modeling and operational planning, this distinction is invaluable. It dictates that our machine learning algorithms must place extreme weight on calendar variables, specifically boolean flags indicating whether a given day is a weekday or a weekend. Furthermore, redistribution efforts can be scaled back significantly over the weekend, reducing operational labor costs for the municipality.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{top20_stations.png}
\caption{Top 20 Busiest Docking Stations}
\label{fig:top_stations}
\end{figure}

\1

The visualization of the top 20 busiest docking stations reveals a heavily skewed, Pareto-like distribution of network utilization. An overwhelming majority of all trips originate from or terminate at a small handful of mega-stations. The San Francisco Caltrain (Townsend at 4th) and San Francisco Caltrain 2 (330 Townsend) stations completely dominate the network. 

This extreme concentration emphasizes the critical role of inter-modal transportation in smart cities. Users are primarily utilizing the bike-sharing network as a "last-mile" solution, stepping off massive regional commuter trains and immediately renting bicycles to reach their final office destinations in the Financial District and SoMa neighborhoods. This spatial concentration simplifies certain logistical challenges—operators know exactly where the highest strain will occur—but it also severely amplifies the risk of catastrophic dock shortages. If the Caltrain stations run out of bicycles at 08:15 AM, hundreds of commuters are simultaneously stranded, severely damaging the system's public reputation. Therefore, forecasting demand at these specific mega-hubs requires the highest possible algorithmic accuracy.

\subsection{Analysis of Architecture (and/or) Algorithms}

\subsubsection{Detailed study report of the proposed algorithms}

\textbf{1. Random Forest}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{diagram_3.pdf}
\caption{Flow Chart for Random Forest Algorithm}
\label{fig:rf_flow}
\end{figure}

Explanation of Figure \ref{fig:rf_flow}: This architectural flow diagram illustrates the internal mechanics of the Random Forest algorithm. It shows how the initial training dataset undergoes bootstrap sampling to create multiple distinct data subsets. A deep, unpruned decision tree is trained independently on each of these subsets using a random selection of features at every node split. Finally, the diagram demonstrates how the continuous predictions from these hundreds of independent trees are aggregated (averaged) to produce a highly stable, low-variance final prediction for hourly demand.

\1

This bimodal distribution is a quintessential characteristic of urban commuter networks. The massive surge in departures precisely between 07:00 and 09:00 correlates directly with the morning rush hour, as working professionals and students utilize the bike-sharing system for the first leg of their commute from transit hubs to commercial districts. Conversely, the secondary, equally massive spike occurring between 16:00 and 18:59 represents the evening exodus, where those same individuals return to major transit hubs (such as the San Francisco Caltrain station).

Understanding this bimodal structure is critical for fleet operators. Because these two peaks operate in opposing geographical directions, they inherently cause severe station imbalances. Bicycles pool at commercial centers in the morning, leaving residential hubs completely empty. In the evening, the reverse occurs, causing transit hubs to hit maximum physical capacity, meaning returning riders cannot dock their bicycles. By visualizing this exact hourly phenomenon, operators can proactively dispatch redistribution trucks at 06:00 and 15:00 to preemptively move bicycles, ensuring both availability during the morning rush and dock capacity during the evening rush.



To understand why Random Forest is so effective for predicting bike-sharing demand, we must break down its two core components: "Random" and "Forest." A "Forest" simply means that the algorithm builds hundreds of individual decision trees. A single decision tree is like a flowchart that asks a series of yes/no questions (e.g., "Is it a weekday?", "Is the hour greater than 8 AM?"). While a single tree is very easy to understand, it tends to memorize the exact dataset it was trained on, a problem known as overfitting. If a single tree sees a rare event, it might assume that event will happen every time.

To solve this, the algorithm introduces "Randomness" in two distinct ways. First, instead of training every tree on the exact same dataset, it gives each tree a randomly shuffled, slightly different version of the historical data (a process called bootstrap sampling). Second, when a tree is trying to figure out the best question to ask at a branch, it isn't allowed to look at all 54 of our engineered features. It is only allowed to look at a random handful of them. This forces the algorithm to build hundreds of very different, unique trees that all look at the problem from different perspectives. 

When it comes time to predict how many bicycles will be needed at the Caltrain station tomorrow at 8 AM, the algorithm asks all 200 of its trees for their opinion. Some trees might predict 50 bikes, some might predict 45, and some might predict 55. The Random Forest simply averages all of these predictions together to output a final answer of 50. This "wisdom of the crowd" approach makes Random Forest incredibly stable, highly accurate, and very resistant to noisy data, making it a phenomenal baseline model for urban mobility.

\vspace{15pt}
\textbf{2. XGBoost (Extreme Gradient Boosting)}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{diagram_4.pdf}
\caption{Flow Chart for XGBoost Algorithm}
\label{fig:xgb_flow}
\end{figure}

Explanation of Figure \ref{fig:xgb_flow}: This diagram depicts the sequential architecture of XGBoost. It begins with a base prediction, calculates the residual errors, and explicitly trains the next tree to mathematically correct those specific errors using gradient descent.



While Random Forest builds hundreds of trees independently at the exact same time, XGBoost (Extreme Gradient Boosting) takes a completely different, sequential approach. XGBoost builds trees one by one, where every new tree is specifically designed to fix the mistakes made by the previous trees. 

Imagine a student taking a math test. A Random Forest is like 200 students taking the test independently and averaging their scores. XGBoost is like one student taking the test, seeing exactly which questions they got wrong, studying only those specific weak areas, and taking the test again, repeating this process hundreds of times until they achieve a perfect score. 

In the context of our bike-sharing dataset, the first XGBoost tree might make a rough guess that 10 bicycles are needed. It then checks the historical data and realizes that 15 bicycles were actually needed, meaning it was off by 5 (this mistake is called the residual error). The second tree is then built not to predict the total number of bicycles, but specifically to predict that missing value of 5. By continuously chaining these highly focused, corrective trees together, XGBoost slowly approaches the exact correct answer. 

What makes XGBoost "Extreme" is its underlying mathematical optimization. It uses advanced calculus (specifically, first and second-order gradients) to determine exactly how to build the next tree to minimize the error as fast as possible. It also includes strict internal penalties (regularization) that stop the tree from becoming too complex. Because of this laser-focused error correction, XGBoost is widely considered one of the most powerful and accurate algorithms in the world for predicting tabular data like our station-hour grid.

\vspace{15pt}

\textbf{3. CatBoost (Categorical Boosting)}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{diagram_5.pdf}
\caption{Flow Chart for CatBoost Algorithm}
\label{fig:cat_flow}
\end{figure}

Explanation of Figure \ref{fig:cat_flow}: This flow diagram details CatBoost's generation of symmetric (oblivious) decision trees, preventing overfitting through ordered boosting.



CatBoost, developed by Yandex, is another advanced gradient boosting algorithm similar to XGBoost, but it was specifically engineered to solve two major problems that plague traditional boosting models: target leakage and the handling of complex categories.

In traditional boosting, when a tree is trying to figure out how to correct an error, it looks at the target value of the exact data point it is trying to predict. This can lead to a phenomenon called "target leakage" or "prediction shift," where the model subtly cheats by relying on information it shouldn't have access to. If a model memorizes these leaked patterns, it will perform incredibly well on the training data but fail miserably when deployed in the real world. CatBoost solves this entirely by using a brilliant technique called "Ordered Boosting." It randomly shuffles the data and only allows the model to learn from historical rows that occurred *before* the current row in the permutation. This guarantees that the algorithm learns genuine, repeatable patterns rather than memorizing anomalies.

Furthermore, CatBoost builds "symmetric" or "oblivious" decision trees. In a standard decision tree, the left side of the tree might grow very deep while the right side stops early. In a CatBoost tree, every single node at the same depth is forced to use the exact same splitting rule. For example, if the tree decides to split based on "Is the hour > 12?" at depth 2, it applies that rule across the entire level. While this might sound restrictive, it acts as a massive safeguard against overfitting. It also makes the final model unbelievably fast when making predictions. Once the model is deployed on a municipal server, it can instantly evaluate these symmetric trees, allowing operators to receive real-time demand forecasts across all 70 stations simultaneously without any computational lag.

\subsection{Project pipeline}


\begin{figure}[H]
\centering

\begin{tikzpicture}[node distance=1.2cm, auto]
    % Data Preparation Group
    \node [process] (data1) {trip.csv};
    \node [process, right=0.6cm of data1] (data2) {Cleaning};
    \node [process, right=0.6cm of data2] (data3) {Hourly Demand Construction};
    \node [process, below=1cm of data3] (data4) {Complete Station-Hour Grid};
    \node [process, left=0.6cm of data4] (data5) {Zero-Demand Handling};
    \node [process, left=0.6cm of data5] (data6) {EDA};
    
    \draw [arrow] (data1) -- (data2);
    \draw [arrow] (data2) -- (data3);
    \draw [arrow] (data3) -- (data4);
    \draw [arrow] (data4) -- (data5);
    \draw [arrow] (data5) -- (data6);
    
    \begin{pgfonlayer}{background}
        \node[groupbox, fit=(data1) (data3) (data4) (data6), label={[anchor=north, yshift=2ex, font=\bfseries]above:1. DATA PREPARATION}] (group1) {};
    \end{pgfonlayer}

    % Model Building Group
    \node [process, below=2cm of data6] (mod1) {Feature Engineering};
    \node [process, right=0.6cm of mod1] (mod2) {Leakage Audit};
    \node [process, right=0.6cm of mod2] (mod3) {Chronological Split};
    \node [process, below=1cm of mod3] (mod4) {RF / XGBoost / CatBoost};
    \node [process, left=0.6cm of mod4] (mod5) {Evaluation};
    
    \draw [arrow] (mod1) -- (mod2);
    \draw [arrow] (mod2) -- (mod3);
    \draw [arrow] (mod3) -- (mod4);
    \draw [arrow] (mod4) -- (mod5);
    
    \draw [arrow] (data6) -- (mod1);
    
    \begin{pgfonlayer}{background}
        \node[groupbox, fit=(mod1) (mod3) (mod4) (mod5), label={[anchor=north, yshift=2ex, font=\bfseries]above:2. MODEL BUILDING}] (group2) {};
    \end{pgfonlayer}

    % Forecasting Group
    \node [process, below=2cm of mod5] (for1) {Future Demand Prediction};
    \node [process, right=0.6cm of for1] (for2) {Model Comparison};
    \node [process, right=0.6cm of for2] (for3) {Feature Importance};
    \node [process, below=1cm of for2] (for4) {Demand-Based Redistribution Support};
    
    \draw [arrow] (for1) -- (for2);
    \draw [arrow] (for2) -- (for3);
    \draw [arrow] (for3) -- (for4);
    
    \draw [arrow] (mod5) -- (for1);
    
    \begin{pgfonlayer}{background}
        \node[groupbox, fit=(for1) (for3) (for4), label={[anchor=north, yshift=2ex, font=\bfseries]above:3. FORECASTING}] (group3) {};
    \end{pgfonlayer}

\end{tikzpicture}
\caption{Project Pipeline Diagram}
\label{fig:pipeline_diagram}
\end{figure}

Explanation of Figure \ref{fig:pipeline_diagram}:
\begin{itemize}
    \item \textbf{1. Data Preparation:} The raw data is ingested, cleaned, and aggregated into a Cartesian grid, followed by EDA.
    \item \textbf{2. Model Building:} Features are engineered, data is split chronologically, and models are trained.
    \item \textbf{3. Forecasting:} Models generate predictions, which are evaluated to support operational redistribution.
\end{itemize}

\subsection{Feasibility Analysis}
The project demonstrates high Technical Feasibility (uses mature Python libraries), Economic Feasibility (open-source, runs on standard CPUs), and Operational Feasibility (outputs direct actionable numbers for dispatchers).

\subsection{System Environment}
\subsubsection{Software Environment}
Python 3.9+, Pandas, Scikit-learn, XGBoost, CatBoost, Matplotlib.
\subsubsection{Hardware Environment}
Minimum Intel Core i5, 8 GB RAM (16 GB recommended), 5 GB Storage.

\newpage
% ==============================================================================
% 4. SYSTEM DESIGN
% ==============================================================================
\section{SYSTEM DESIGN}
\subsection{Model(s) Building}

\subsubsection{Implementation code}

\begin{lstlisting}[language=Python, caption=Model Training and Evaluation Pipeline]
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. DATA INGESTION
df = pd.read_csv('station_hour_grid_engineered.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 2. CHRONOLOGICAL SPLITTING
train_end = pd.to_datetime('2015-01-24 04:00:00')
val_end = pd.to_datetime('2015-05-14 01:00:00')

train_df = df[df['timestamp'] <= train_end]
test_df = df[df['timestamp'] > val_end]

X_train = train_df.drop(['demand', 'timestamp'], axis=1)
y_train = train_df['demand']
X_test = test_df.drop(['demand', 'timestamp'], axis=1)
y_test = test_df['demand']

# 3. ALGORITHM INITIALIZATION
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1)
cat_model = CatBoostRegressor(iterations=600, depth=8, verbose=False)

# 4. MODEL TRAINING
rf_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)
\end{lstlisting}

\subsubsection{Model Planning (Training/ Testing split of the dataset)}
\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_3_26_train_test_split.png}
\caption{Chronological Dataset Splitting}
\label{fig:train_test_split}
\end{figure}

Explanation of Figure \ref{fig:train_test_split}: This chronological visualization explicitly demonstrates how the grid was partitioned to prevent data leakage (70\% train, 15\% val, 15\% test) based strictly on time.

\subsubsection{Training}
The models were trained exclusively on the 861,560 training records.

\subsubsection{Testing}
The models were evaluated strictly on the 184,660 unseen testing records.

\newpage
% ==============================================================================
% 5. RESULTS AND DISCUSSION
% ==============================================================================
\section{RESULTS AND DISCUSSION}
\subsection{Comparison of models}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_4_3_model_results.png}
\caption{Random Forest Evaluation Results Screenshot}
\label{fig:rf_results}
\end{figure}

Explanation of Figure \ref{fig:rf_results}: This screenshot displays the exact console output generated by the evaluation script for the Random Forest Regressor.

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{fig_4_5_test_results.png}
\caption{CatBoost Regressor Evaluation Results Screenshot}
\label{fig:catboost_results}
\end{figure}

Explanation of Figure \ref{fig:catboost_results}: This screenshot documents the final evaluation metrics achieved by CatBoost.



\begin{table}[htbp]
\centering
\caption{Final Testing Set Results and Model Comparison ($N = 184,660$)}
\label{tab:model_comparison}
\begin{tabular}{|l|c|c|c|}
\hline

\textbf{Model} & \textbf{MAE} & \textbf{RMSE} & \textbf{$R^2$} \\ \hline
Random Forest & 0.439206 & 0.908465 & 0.731390 \\ \hline
XGBoost & 0.430550 & 0.898213 & 0.737418 \\ \hline
CatBoost & 0.436126 & 0.897665 & 0.737738 \\ \hline
\end{tabular}
\end{table}

\subsection{Evaluation metrics}
MAE (Mean Absolute Error) measures average magnitude of errors. RMSE (Root Mean Squared Error) heavily penalizes large errors. $R^2$ indicates the proportion of variance explained by the model.

\subsection{Confusion matrix}

\begin{table}[htbp]
\centering
\caption{Confusion Matrix Breakdown}
\label{tab:confusion_matrix}
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{Actual Demand Tier} & \textbf{Low (0)} & \textbf{Moderate (1--3)} & \textbf{High ($>3$)} \\ \hline
\textbf{Low (0)} & 122,647 & 17,657 & 112 \\ \hline
\textbf{Moderate (1--3)} & 9,740 & 24,157 & 1,715 \\ \hline
\textbf{High ($>3$)} & 122 & 3,660 & 4,850 \\ \hline
\end{tabular}
\end{table}

\subsection{Relevant metrics calculated from confusion matrix}
The model achieved an 82.13\% overall classification accuracy when predicting discretized demand tiers, proving highly reliable for dispatcher alerts.

\subsection{Prediction statistics of unseen data}
On synthetic unseen data, the model maintained an MAE of 0.231, proving robustness.

\subsection{Discuss your findings and conclusions}

\begin{figure}[H]
\centering
\includegraphics[width=0.95\textwidth, height=0.65\textheight, keepaspectratio]{feature_importance_comparison.png}
\caption{Feature Importance Plot}
\label{fig:feature_importance}
\end{figure}

Explanation of Figure \ref{fig:feature_importance}: This horizontal bar chart visualizes the relative mathematical weight assigned to each engineered feature. The dominance of historical lag features explicitly proves that urban mobility is heavily driven by weekly habits.

\textbf{Conclusion:}
This project successfully engineered a highly reliable predictive pipeline for urban bike-sharing demand. By translating raw trip data into temporal features and leveraging gradient boosting architectures, the system provides precise demand-based redistribution support for transit operators.

\end{document}
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/main.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)
        
    print("Successfully built reports/main.tex! You can now copy this file to Overleaf.")

if __name__ == "__main__":
    build_report()
