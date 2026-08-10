import re

with open('main.tex', 'r') as f:
    content = f.read()

# 1. Update the mapping sentence
old_mapping = r'We map the physical planar coordinates of the $p$-th sequential body to the local state vectors: translational position $\mathbf{x}_{p} = [x_{6p-5},\, x_{6p-4}]^T$, translational velocity $\dot{\mathbf{x}}_{p} = [x_{6p-2},\, x_{6p-1}]^T$, rotational position $x_{p} = x_{6p-3}$, and rotational velocity $\dot{x}_{p} = x_{6p}$.'
new_mapping = r'We map the physical planar coordinates of the $p$-th sequential body to the local state vectors: translational position $\mathbf{x}_{2p-1} = [x_{6p-5},\, x_{6p-4}]^T$, translational velocity $\dot{\mathbf{x}}_{2p-1} = [x_{6p-2},\, x_{6p-1}]^T$, rotational position $\mathbf{x}_{2p} = x_{6p-3}$, and rotational velocity $\dot{\mathbf{x}}_{2p} = x_{6p}$.'

content = content.replace(old_mapping, new_mapping)

# 2. Extract the equation sections
# Section 1: General State-Space Form for the Chain Bodies
# Section 2: Appendix: State-Space Equations for the Wafer Chain

def process_equations(text):
    # Change translational vectors: \mathbf{x}_p -> \mathbf{x}_{2p-1}
    # For expressions like \mathbf{x}_{p}, \mathbf{x}_{p+1}, \mathbf{x}_{p-1}
    
    # We will do regex substitution carefully.
    # We need to handle \mathbf{x}_{p}, \mathbf{x}_{1}, \mathbf{x}_{p_{2_k}}, etc.
    
    # It's easier to write custom functions for each body equation.
    return text

# We will just rewrite the equations part using Python formatting

general_eqs = r"""
For the \textbf{first body in each chain} ($i=2$, index $p$, upstream index $1$), the translational state derivatives ($\ddot{\mathbf{x}}_{2p-1} = \ddot{f}_{2_j}$) and rotational state derivatives ($\ddot{\mathbf{x}}_{2p} = \ddot{\theta}_{z_{2_j}}$) take the form:
\begin{align}
    \ddot{\mathbf{x}}_{2p-1} &= 
    \frac{1}{\bar{m}_{2_j}} \Big[ u^0_{2_j} - {}^0\bar{C}_{2_j}\dot{\mathbf{x}}_{2p-1} - {}^0\bar{K}_{2_j}\mathbf{x}_{2p-1} + {}^0\hat{C}_{3_j}\dot{\mathbf{x}}_{2p+1} + {}^0\hat{K}_{3_j}\mathbf{x}_{2p+1} \Big] \notag \\
    &\quad - \frac{1}{m_0} \Big[ u^0_1 - {}^0\bar{C}_1\dot{\mathbf{x}}_{1} - {}^0\bar{K}_1 \mathbf{x}_{1} + \sum_{k=1}^3 \left( {}^0\hat{C}_{2_k}\dot{\mathbf{x}}_{2p_{2_k}-1} + {}^0\hat{K}_{2_k}\mathbf{x}_{2p_{2_k}-1} \right) \Big] \\
    \ddot{\mathbf{x}}_{2p} &= 
    \frac{1}{I_{z_{2_j}}} \Big[ \tau^0_{2_j} - {}^0\bar{c}_{\theta,2_j}\dot{\mathbf{x}}_{2p} - {}^0\bar{k}_{\theta,2_j}\mathbf{x}_{2p} + {}^0\hat{c}_{\theta,3_j}\dot{\mathbf{x}}_{2p+2} + {}^0\hat{k}_{\theta,3_j}\mathbf{x}_{2p+2} \Big] \notag \\
    &\quad - \frac{1}{I_{z_0}} \Big[ \tau^0_1 - {}^0\bar{C}_1\dot{\mathbf{x}}_{2} - {}^0\bar{K}_1 \mathbf{x}_{2} + \sum_{k=1}^3 \left( {}^0\hat{c}_{\theta,2_k}\dot{\mathbf{x}}_{2p_{2_k}} + {}^0\hat{k}_{\theta,2_k}\mathbf{x}_{2p_{2_k}} \right) \Big]
\end{align}

For a \textbf{general intermediate body} ($2 < i < N_j$, index $p$), the dynamics are shaped by upstream, local, and downstream interactions:
\begin{align}
    \ddot{\mathbf{x}}_{2p-1} &= 
    \frac{1}{\bar{m}_{i_j}} \Big[ u^0_{i_j} - {}^0\bar{C}_{i_j}\dot{\mathbf{x}}_{2p-1} - {}^0\bar{K}_{i_j}\mathbf{x}_{2p-1} + {}^0\hat{C}_{(i+1)_j}\dot{\mathbf{x}}_{2p+1} + {}^0\hat{K}_{(i+1)_j}\mathbf{x}_{2p+1} \Big] \notag \\
    &\quad - \frac{1}{\bar{m}_{(i-1)_j}} \Big[ u^0_{(i-1)_j} - {}^0\bar{C}_{(i-1)_j}\dot{\mathbf{x}}_{2p-3} - {}^0\bar{K}_{(i-1)_j}\mathbf{x}_{2p-3} + {}^0\hat{C}_{i_j}\dot{\mathbf{x}}_{2p-1} + {}^0\hat{K}_{i_j}\mathbf{x}_{2p-1} \Big] \\
    \ddot{\mathbf{x}}_{2p} &= 
    \frac{1}{I_{z_{i_j}}} \Big[ \tau^0_{i_j} - {}^0\bar{c}_{\theta,i_j}\dot{\mathbf{x}}_{2p} - {}^0\bar{k}_{\theta,i_j}\mathbf{x}_{2p} + {}^0\hat{c}_{\theta,(i+1)_j}\dot{\mathbf{x}}_{2p+2} + {}^0\hat{k}_{\theta,(i+1)_j}\mathbf{x}_{2p+2} \Big] \notag \\
    &\quad - \frac{1}{I_{z_{(i-1)_j}}} \Big[ \tau^0_{(i-1)_j} - {}^0\bar{c}_{\theta,(i-1)_j}\dot{\mathbf{x}}_{2p-2} - {}^0\bar{k}_{\theta,(i-1)_j}\mathbf{x}_{2p-2} + {}^0\hat{c}_{\theta,i_j}\dot{\mathbf{x}}_{2p} + {}^0\hat{k}_{\theta,i_j}\mathbf{x}_{2p} \Big]
\end{align}

Finally, for the \textbf{end-effector body} ($i = N_j$, index $p$), the downstream force terms naturally vanish as there are no subsequent bodies:
\begin{align}
    \ddot{\mathbf{x}}_{2p-1} &= 
    \frac{1}{\bar{m}_{{N_j}_j}} \Big[ u^0_{{N_j}_j} - {}^0\bar{C}_{{N_j}_j}\dot{\mathbf{x}}_{2p-1} - {}^0\bar{K}_{{N_j}_j}\mathbf{x}_{2p-1} \Big] \notag \\
    &\quad - \frac{1}{\bar{m}_{({N_j}-1)_j}} \Big[ u^0_{({N_j}-1)_j} - {}^0\bar{C}_{({N_j}-1)_j}\dot{\mathbf{x}}_{2p-3} - {}^0\bar{K}_{({N_j}-1)_j}\mathbf{x}_{2p-3} + {}^0\hat{C}_{{N_j}_j}\dot{\mathbf{x}}_{2p-1} + {}^0\hat{K}_{{N_j}_j}\mathbf{x}_{2p-1} \Big] \\
    \ddot{\mathbf{x}}_{2p} &= 
    \frac{1}{I_{z_{{N_j}_j}}} \Big[ \tau^0_{{N_j}_j} - {}^0\bar{c}_{\theta,{N_j}_j}\dot{\mathbf{x}}_{2p} - {}^0\bar{k}_{\theta,{N_j}_j}\mathbf{x}_{2p} \Big] \notag \\
    &\quad - \frac{1}{I_{z_{({N_j}-1)_j}}} \Big[ \tau^0_{({N_j}-1)_j} - {}^0\bar{c}_{\theta,({N_j}-1)_j}\dot{\mathbf{x}}_{2p-2} - {}^0\bar{k}_{\theta,({N_j}-1)_j}\mathbf{x}_{2p-2} + {}^0\hat{c}_{\theta,{N_j}_j}\dot{\mathbf{x}}_{2p} + {}^0\hat{k}_{\theta,{N_j}_j}\mathbf{x}_{2p} \Big]
\end{align}
"""

start_idx = content.find(r'For the \textbf{first body in each chain}')
end_idx = content.find(r'\section*{Appendix: State-Space Equations for the Wafer Chain}')

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + general_eqs + content[end_idx:]

# Process appendix
appendix_text = r"""\section*{Appendix: State-Space Equations for the Wafer Chain}

The wafer chain corresponds to $j=3$, with $N_3=8$ (bodies $i=2, \ldots, 8$). These map to global state vectors $\mathbf{x}_{2p-1}$ (translational) and $\mathbf{x}_{2p}$ (rotational) for $p=12, \ldots, 18$. The state-space equations are represented below.

\textbf{First body of the wafer chain ($i=2$, $p=12$):}
\begin{align}
    \ddot{\mathbf{x}}_{23} &= 
    \frac{1}{\bar{m}_{2_3}} \Big[ u^0_{2_3} - {}^0\bar{C}_{2_3}\dot{\mathbf{x}}_{23} - {}^0\bar{K}_{2_3}\mathbf{x}_{23} + {}^0\hat{C}_{3_3}\dot{\mathbf{x}}_{25} + {}^0\hat{K}_{3_3}\mathbf{x}_{25} \Big] \notag \\
    &\quad - \frac{1}{m_0} \Big[ u^0_1 - {}^0\bar{C}_1\dot{\mathbf{x}}_{1} - {}^0\bar{K}_1 \mathbf{x}_{1} + \sum_{k=1}^3 \left( {}^0\hat{C}_{2_k}\dot{\mathbf{x}}_{2p_{2_k}-1} + {}^0\hat{K}_{2_k}\mathbf{x}_{2p_{2_k}-1} \right) \Big] \\
    \ddot{\mathbf{x}}_{24} &= 
    \frac{1}{I_{z_{2_3}}} \Big[ \tau^0_{2_3} - {}^0\bar{c}_{\theta,2_3}\dot{\mathbf{x}}_{24} - {}^0\bar{k}_{\theta,2_3}\mathbf{x}_{24} + {}^0\hat{c}_{\theta,3_3}\dot{\mathbf{x}}_{26} + {}^0\hat{k}_{\theta,3_3}\mathbf{x}_{26} \Big] \notag \\
    &\quad - \frac{1}{I_{z_0}} \Big[ \tau^0_1 - {}^0\bar{C}_1\dot{\mathbf{x}}_{2} - {}^0\bar{K}_1 \mathbf{x}_{2} + \sum_{k=1}^3 \left( {}^0\hat{c}_{\theta,2_k}\dot{\mathbf{x}}_{2p_{2_k}} + {}^0\hat{k}_{\theta,2_k}\mathbf{x}_{2p_{2_k}} \right) \Big]
\end{align}
"""

for i in range(3, 8):
    p = 10 + i
    t_idx = 2*p - 1
    r_idx = 2*p
    t_next = t_idx + 2
    r_next = r_idx + 2
    t_prev = t_idx - 2
    r_prev = r_idx - 2
    
    appendix_text += f"""
\\textbf{{Intermediate body ($i={i}$, $p={p}$):}}
\\begin{{align}}
    \\ddot{{\\mathbf{{x}}}}_{{{t_idx}}} &= 
    \\frac{{1}}{{\\bar{{m}}_{{{i}_3}}}} \\Big[ u^0_{{{i}_3}} - {{}}^0\\bar{{C}}_{{{i}_3}}\\dot{{\\mathbf{{x}}}}_{{{t_idx}}} - {{}}^0\\bar{{K}}_{{{i}_3}}\\mathbf{{x}}_{{{t_idx}}} + {{}}^0\\hat{{C}}_{{({i+1})_3}}\\dot{{\\mathbf{{x}}}}_{{{t_next}}} + {{}}^0\\hat{{K}}_{{({i+1})_3}}\\mathbf{{x}}_{{{t_next}}} \\Big] \\notag \\\\
    &\\quad - \\frac{{1}}{{\\bar{{m}}_{{({i-1})_3}}}} \\Big[ u^0_{{({i-1})_3}} - {{}}^0\\bar{{C}}_{{({i-1})_3}}\\dot{{\\mathbf{{x}}}}_{{{t_prev}}} - {{}}^0\\bar{{K}}_{{({i-1})_3}}\\mathbf{{x}}_{{{t_prev}}} + {{}}^0\\hat{{C}}_{{{i}_3}}\\dot{{\\mathbf{{x}}}}_{{{t_idx}}} + {{}}^0\\hat{{K}}_{{{i}_3}}\\mathbf{{x}}_{{{t_idx}}} \\Big] \\\\
    \\ddot{{\\mathbf{{x}}}}_{{{r_idx}}} &= 
    \\frac{{1}}{{I_{{z_{{{i}_3}}}}}} \\Big[ \\tau^0_{{{i}_3}} - {{}}^0\\bar{{c}}_{{\\theta,{i}_3}}\\dot{{\\mathbf{{x}}}}_{{{r_idx}}} - {{}}^0\\bar{{k}}_{{\\theta,{i}_3}}\\mathbf{{x}}_{{{r_idx}}} + {{}}^0\\hat{{c}}_{{\\theta,({i+1})_3}}\\dot{{\\mathbf{{x}}}}_{{{r_next}}} + {{}}^0\\hat{{k}}_{{\\theta,({i+1})_3}}\\mathbf{{x}}_{{{r_next}}} \\Big] \\notag \\\\
    &\\quad - \\frac{{1}}{{I_{{z_{{({i-1})_3}}}}}} \\Big[ \\tau^0_{{({i-1})_3}} - {{}}^0\\bar{{c}}_{{\\theta,({i-1})_3}}\\dot{{\\mathbf{{x}}}}_{{{r_prev}}} - {{}}^0\\bar{{k}}_{{\\theta,({i-1})_3}}\\mathbf{{x}}_{{{r_prev}}} + {{}}^0\\hat{{c}}_{{\\theta,{i}_3}}\\dot{{\\mathbf{{x}}}}_{{{r_idx}}} + {{}}^0\\hat{{k}}_{{\\theta,{i}_3}}\\mathbf{{x}}_{{{r_idx}}} \\Big]
\\end{{align}}
"""

appendix_text += r"""
\textbf{End-effector body ($i=8$, $p=18$):}
\begin{align}
    \ddot{\mathbf{x}}_{35} &= 
    \frac{1}{\bar{m}_{8_3}} \Big[ u^0_{8_3} - {}^0\bar{C}_{8_3}\dot{\mathbf{x}}_{35} - {}^0\bar{K}_{8_3}\mathbf{x}_{35} \Big] \notag \\
    &\quad - \frac{1}{\bar{m}_{7_3}} \Big[ u^0_{7_3} - {}^0\bar{C}_{7_3}\dot{\mathbf{x}}_{33} - {}^0\bar{K}_{7_3}\mathbf{x}_{33} + {}^0\hat{C}_{8_3}\dot{\mathbf{x}}_{35} + {}^0\hat{K}_{8_3}\mathbf{x}_{35} \Big] \\
    \ddot{\mathbf{x}}_{36} &= 
    \frac{1}{I_{z_{8_3}}} \Big[ \tau^0_{8_3} - {}^0\bar{c}_{\theta,8_3}\dot{\mathbf{x}}_{36} - {}^0\bar{k}_{\theta,8_3}\mathbf{x}_{36} \Big] \notag \\
    &\quad - \frac{1}{I_{z_{7_3}}} \Big[ \tau^0_{7_3} - {}^0\bar{c}_{\theta,7_3}\dot{\mathbf{x}}_{34} - {}^0\bar{k}_{\theta,7_3}\mathbf{x}_{34} + {}^0\hat{c}_{\theta,8_3}\dot{\mathbf{x}}_{36} + {}^0\hat{k}_{\theta,8_3}\mathbf{x}_{36} \Big]
\end{align}
"""

start_app = content.find(r'\section*{Appendix: State-Space Equations for the Wafer Chain}')
end_app = content.find(r'\printbibliography')

if start_app != -1 and end_app != -1:
    content = content[:start_app] + appendix_text + '\n' + content[end_app:]

with open('main.tex', 'w') as f:
    f.write(content)

