clear all
clc

% Three-point bending test on an unnotched beam with phase field damage 
% Force controlled test
% Staggered iterative solution with force increments up to failure
% 
% Marco Paggi (2025) Mechanics of complex network materials: a formulation
% based on phase field damage evolution on graphs,
% Computer Methods in Applied Mechanics and Engineering, in press.
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   INPUT SECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    cases=1; % 1: bilateral damage; 2: monolateral damage

    % Beam geometrical parameters
    L_beam = 50.8e-3;          % Beam span [m]
    h = 3.18e-3;               % Cross-section height [m]
    b = 12.7e-3;               % Cross-section width [m]  
    
    % Material parameters
    E0 = 1826e6;               % Young's modulus [Pa]
    Gc = 2e-1;                 % [Pa sqrt(m)]
    l0 = 4.0e-3;               % Internal length scale [m]     
    k  = 0.001;                % residual stiffness
    H_c= 0.0;                  % threshold driving force term

    % Force control parameters
    F_max = 160;               % Maximum force to be applied [N]
    n_force_steps = 60;        % Number of loading steps
    F_values = linspace(0, F_max, n_force_steps);    
        
    % Discretization and numerical parameters
    n_elements = 300;          % Number of beam subdivisions
    max_staggered_iter = 100;  % Maximum number of staggered iterations
    tol_s = 1e-6;              % Convergence tolerance for s
    relaxation = 0.5;          % Relaxation factor for the update of s 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   END INPUT SECTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
    % Discretization
    dz = L_beam / n_elements;
    z_nodes = linspace(0, L_beam, n_elements + 1)';
    n_nodes = length(z_nodes);
    
    % Find mid-span node
    [~, mid_node] = min(abs(z_nodes - L_beam/2));
    
    % Initialize storage for results
    deflection_midspan = zeros(n_force_steps, 1);
    max_damage = zeros(n_force_steps, 1);
    curvature_midspan = zeros(n_force_steps, 1);
    s_history = zeros(n_nodes, n_force_steps);
    chi_history = zeros(n_nodes, n_force_steps);
    deflection_history = zeros(n_nodes, n_force_steps);
    
    % Initial state (undamaged)
    s = zeros(n_nodes, 1);
    
    fprintf('=== FORCE-CONTROLLED THREE-POINT BENDING TEST ===\n');
    fprintf('Beam: L=%.2fm, h=%.3fm, b=%.3fm\n', L_beam, h, b);
    fprintf('Force: %.0f to %.0f N (%d steps)\n', F_values(1), F_values(end), n_force_steps);
    fprintf('Material: E0=%.2e Pa, Gc=%.1f N/m, l0=%.3f m, k=%.3f\n', E0, Gc, l0, k);
    fprintf('Discretization: %d elements, %d nodes\n\n', n_elements, n_nodes);
    
    norma=[];

    % Force-controlled loop
    for step = 1:n_force_steps
        F_current = F_values(step);
        fprintf('Force step %d/%d: F = %.1f N\n', step, n_force_steps, F_current);
        
        % Staggered iteration loop for the current force level
        s_converged = s;  % Start from previous converged state
        
        for iter = 1:max_staggered_iter
            % Store previous s for convergence check
            s_prev = s_converged;           

            % STEP 1: Solve mechanical problem with the current s field
            [eps0_vec, chi_vec, M_vec, N_vec] = solve_mechanical_problem(z_nodes, s_converged, F_current, E0, b, h, L_beam, k, cases);
            
            % STEP 2: Compute energy density psi at each node
            psi_vec = compute_energy_density(M_vec, N_vec, eps0_vec, chi_vec);
                   
            % STEP 3: Solve damage evolution equation 
            s_new = solve_damage_problem_Dirichlet(psi_vec,  Gc, l0, dz, n_nodes, H_c);          
            
            % Relaxed update
            s_converged = relaxation * s_new + (1 - relaxation) * s_prev;
            
            % Convergence check
            s_norm_change = norm(s_converged - s_prev) / norm(s_prev + eps);

            norma(step,iter)=s_norm_change;
            
            if s_norm_change < tol_s
                fprintf('  Staggered converged in %d iterations (||Δs|| = %.2e)\n', iter, s_norm_change);
                break;
            end
            
            if iter == max_staggered_iter
                fprintf('  Max staggered iterations reached (||Δs|| = %.2e)\n', s_norm_change);
            end
        end  % end loop over staggered iterations
        
        % Update damage field for the next force step
        s = s_converged;
        
        % STEP 4: Compute deflection by integrating the curvature
        deflection = compute_deflection(chi_vec, z_nodes, L_beam);
        deflection_midspan(step) = deflection(mid_node);
        
        % Store results
        max_damage(step) = max(s);
        curvature_midspan(step) = chi_vec(mid_node);
        s_history(:, step) = s;
        chi_history(:, step) = chi_vec;
        deflection_history(:, step) = deflection;
        
        % Check for failure (damage close to 1 at any point)
        if max(s) > 0.95
            fprintf('*** FAILURE DETECTED at force step %d (max damage = %.3f) ***\n', step, max(s));
            % Stop the simulation
            deflection_midspan = deflection_midspan(1:step);
            max_damage = max_damage(1:step);
            curvature_midspan = curvature_midspan(1:step);
            F_values = F_values(1:step);
            s_history = s_history(:, 1:step);
            chi_history = chi_history(:, 1:step);
            deflection_history = deflection_history(:, 1:step);
            break;
        end
    end  % end loop over force steps
    
    % Post-processing and plotting
    post_process_results(cases, z_nodes, s_history, deflection_midspan, F_values, max_damage, ...
                        curvature_midspan, b, h, E0, L_beam, chi_history, deflection_history, norma);





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                            FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [eps0_vec, chi_vec, M_vec, N_vec] = solve_mechanical_problem(z_nodes, s, F, E0, b, h, L_beam, k, cases)

    % Solve mechanical problem for a given damage field s 
    n_nodes = length(z_nodes);
    
    % Compute bending moment distribution 
    M_vec = zeros(n_nodes, 1);
    for i = 1:n_nodes
        z = z_nodes(i);
        if z <= L_beam/2
            M_vec(i) = (F/2) * z;  % Left half of the beam
            N_vec(i) = 0;
        else
            M_vec(i) = (F/2) * (L_beam - z);  % Right half of the beam
            N_vec(i) = 0;
        end
    end
    
    % Initialize outputs
    eps0_vec = zeros(n_nodes, 1);
    chi_vec = zeros(n_nodes, 1);
    
    % Solve cross-section equilibrium at each node
    for i = 1:n_nodes
        M_target = M_vec(i);
        N_target = N_vec(i); 
        s_local  = s(i);
                
        if(cases==1) % bilateral damage (linear problem)        
         [eps0_sol, chi_sol] = solve_cross_section1(N_target, M_target, s_local, E0, b, h, k);
        end
        if(cases==2) % monolateral damage (nonlinear problem)
         [eps0_sol, chi_sol] = solve_cross_section2(N_target, M_target, s_local, E0, b, h, k);
        end       
        
        eps0_vec(i) = eps0_sol;
        chi_vec(i) = chi_sol;
    end

    chi_vec(1)=0;  
    chi_vec(n_nodes)=0; 
end


function [eps0_sol, chi_sol] = solve_cross_section1(N_target, M_target, s_local, E0, b, h, k)
    % Solve cross-section equilibrium equations 
    % for given internal reactions and damage s 
        
    A = b * h;                  % Area
    I = b * h^3 / 12;           % Moment of inertia
    Es = E0*((1-s_local)^2+k);  % Damaged Young's modulus
    
    eps0_sol = N_target/(Es*A);
    chi_sol = M_target/(Es*I);
end


function [eps0_sol, chi_sol] = solve_cross_section2(N_target, M_target, s_local, E0, b, h, k)
    err2=1;
    ysteps=500;
    Es=E0*((1-s_local)^2+k);
	minerr=1e20;
    for j=1:ysteps   % j steps to find the value of the neutral axis position that is solving the nonlinear algebraic equation (can be changed by a Newton-Raphson algorithm, or a bisection algorithm)           
        yb=-h/2*(1-(j-1)/(ysteps-1));
        A0=(h/2+yb)*b;
        As=b*h-A0;
        SxA0=-(h^2/4-yb^2)*b/2;
        SxAs=-SxA0;
        IxA0=b*(h/2+yb)*(h/4-yb/2)^2+b/12*(h/2+yb)^3;
        IxAs=b*h^3/12-IxA0;
        M11=(A0*E0+As*Es);
        M12=(SxA0*E0+SxAs*Es);
        M21=M12;
        M22=(IxA0*E0+IxAs*Es);
        M=[M11 M12; M21 M22];
        err1(j)=abs((M22-M12*yb)*N_target-(M12-M11*yb)*M_target);
        if(err1(j)<minerr)
           minerr=err1(j);                 
           ybopt=yb;
        end
    end

    % Compute the solution corresponding to the sought neutral axis position coordinate, yb 

    A0=(h/2+ybopt)*b;
    As=b*h-A0;
    SxA0=-(h^2/4-ybopt^2)*b/2;
    SxAs=-SxA0;
    IxA0=b*(h/2+ybopt)*(h/4-ybopt/2)^2+b/12*(h/2+ybopt)^3;
    IxAs=b*h^3/12-IxA0;
    M11=(A0*E0+As*Es);
    M12=(SxA0*E0+SxAs*Es);
    M21=M12;
    M22=(IxA0*E0+IxAs*Es);
    M=[M11 M12; M21 M22];     
	V=[N_target M_target]';             
    x=inv(M)*V;   

    eps0_sol=x(1);
    chi_sol=x(2);
end


function psi_vec = compute_energy_density(M_vec, N_vec, eps0_vec, chi_vec) 
    % Compute energy density psi at each node
    n_nodes = length(eps0_vec);
    psi_vec = zeros(n_nodes, 1);

    for i = 1:n_nodes
        eps0 = eps0_vec(i);
        chi = chi_vec(i);
        
        if abs(chi) < 1e-12
            psi_vec(i) = 0;
            continue;
        end
        
        psi_vec(i)=0.5*(M_vec(i)*chi_vec(i)+N_vec(i)*eps0_vec(i));
        
        % Ensure non-negative energy
        psi_vec(i) = max(psi_vec(i), 0);
    end
end


function s_new = solve_damage_problem_Dirichlet(psi_vec, Gc, l0, dz, n_nodes, H_c)
    % Solve damage evolution equation using finite differences
    % (H + 1)s - l0^2 * d²s/dz² = H, where H = 2*l0*psi/Gc
    % with Dirichlet BCs: s(0) = 0, s(L) = 0
        
    % Compute H field
    H_vec = 2 * l0 * psi_vec / Gc;

    H_vec(H_vec<H_c)=0; % impose threshold to the driving force
    
    % Finite difference matrix for d²/dz² (central differences)
    A = zeros(n_nodes, n_nodes);
    
    % Interior points (standard central difference)
    for i = 2:n_nodes-1
        A(i, i-1) = 1/dz^2;
        A(i, i) = -2/dz^2;
        A(i, i+1) = 1/dz^2;
    end
    
    % Boundary conditions: s(0) = 0 and s(L) = 0
    % Left boundary: s(1) = 0
    A(1, 1) = 1;
    A(1, 2:end) = 0;
    
    % Right boundary: s(end) = 0
    A(n_nodes, n_nodes) = 1;
    A(n_nodes, 1:end-1) = 0;
    
    % Construct system matrix: (H + 1)I - l0^2 * A
    system_matrix = diag(H_vec + 1) - l0^2 * A;
    
    % Right-hand side - modify for Dirichlet BCs
    rhs = H_vec;
    rhs(1) = 0;      % s(0) = 0
    rhs(end) = 0;    % s(L) = 0
    
    % Solve linear system
    s_new = system_matrix \ rhs;
    
    % Enforce bounds: 0 <= s <= 1 and ensure BCs are exactly satisfied
    s_new = max(min(s_new, 1), 0);
    s_new(1) = 0;    % Ensure exact BC at left
    s_new(end) = 0;  % Ensure exact BC at right
end



function deflection = compute_deflection(chi_vec, z_nodes, L_beam)
    % Compute deflection by double integration of curvature
    % chi = d²v/dz², with boundary conditions v(0) = v(L) = 0
    
    n_nodes = length(z_nodes);
    dz = z_nodes(2) - z_nodes(1);
    
    % First integration: slope theta = dv/dz
    theta = zeros(n_nodes, 1);
    
    % Numerical integration using trapezoidal rule
    for i = 2:n_nodes
        theta(i) = theta(i-1) + 0.5 * dz * (chi_vec(i-1) + chi_vec(i));
    end
    
    % Apply boundary condition: average slope adjustment to satisfy v(L) = 0
    % This is equivalent to making the integral of theta dz from 0 to L equal to 0
    theta_avg = trapz(z_nodes, theta) / L_beam;
    theta = theta - theta_avg;
    
    % Second integration: deflection v
    deflection = zeros(n_nodes, 1);
    for i = 2:n_nodes
        deflection(i) = deflection(i-1) + 0.5 * dz * (theta(i-1) + theta(i));
    end
    
    % Apply boundary condition v(0) = 0 (already satisfied)
    % Apply boundary condition v(L) = 0 by subtracting linear ramp
    deflection = deflection - (z_nodes / L_beam) * deflection(end);
    
    % Debug output to verify deflection is non-zero
    max_deflection = max(abs(deflection));
    if max_deflection < 1e-10
        warning('Very small deflection detected: %.2e. Check curvature values.', max_deflection);
    end
end



function post_process_results(cases,z_nodes, s_history, deflection_midspan, F_values, max_damage, ...
                            curvature_midspan, b, h, E0, L_beam, chi_history, deflection_history, norma)

    n_steps = length(F_values);
    final_step = n_steps;

    Inertia=b*h^3/12;
    delta_y_elastica = F_values* L_beam^3 / (48 * E0 * Inertia);
    
    
    % Plot 1: Force vs Deflection curve of the the methods, with different
    % colors
    figure(1)
    if(cases==1)
       plot(abs(deflection_midspan), F_values, 'k-', 'LineWidth', 2);   
       hold on     
    end    
    if(cases==2)
      plot(abs(deflection_midspan), F_values, 'r-', 'LineWidth', 2);   
      hold on
      plot(delta_y_elastica, F_values, 'k--', 'LineWidth', 2);
    end

    xlabel('Mid-span deflection [m]');
    ylabel('Applied force F [N]');
    xlim([0,0.012])
    
    figure(1)
    hold on
    experiments=[0.0	0
    0.805 20
    1.252 30
    1.700 40
    2.075 50
    2.450 60
    2.925 70
    3.400 80
    4.000 90
    4.700 100
    6.000 109
    7.565 111
    9.130 112];
    experiments(:,1)=experiments(:,1)/1000;     % deflection values converted from mm to m   
    plot(experiments(:,1),experiments(:,2),'ok')        
    
    
    % Plot 2: Damage evolution along the beam for selected load steps 
    % (to be changed as preferred)
    figure(2)    
    if(cases==1)
       selected_steps = [10, 20, 30, 34]; 
    end    
    if(cases==2)
       selected_steps = [10, 20, 30, 34, 40, 46]; 
    end

    colors = lines(length(selected_steps));
    hold on;
    for i = 1:length(selected_steps)
        step = selected_steps(i);
        if(cases==1)
             plot(z_nodes/L_beam, s_history(:, step), 'Color', colors(i, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('F = %.0f N', F_values(step)));
        end
        if(cases==2)
             plot(z_nodes/L_beam, s_history(:, step), '--', 'Color', colors(i, :), 'LineWidth', 2, ...
             'DisplayName', sprintf('F = %.0f N', F_values(step)));
        end        
    end
    xlabel('Beam coordinate z/L [-]');
    ylabel('Damage field s');
    legend('show', 'Location', 'best');
    
    
    % Plot 3: Maximum damage vs Force
    figure(3)
    if(cases==1)
        plot(F_values, max_damage, 'k-', 'LineWidth', 2);
    end
    xlabel('Applied force F [N]');
    ylabel('Maximum damage s_{max}');
    hold on
    if(cases==2)
        plot(F_values, max_damage, 'r-', 'LineWidth', 2);
    end


    % Plot 4: loglog plot of the norm of the error vs. iterations, for the
    % chosen load steps
    if (cases==2) 
    figure(4)
    loglog(norma(33,:),'k')
    hold on
    loglog(norma(40,:)','b')
    loglog(norma(46,:)','r')
    end
    

    % Display key results
    fprintf('\n=== FINAL RESULTS (UPDATED MODEL) ===\n');
    fprintf('Maximum force reached: %.1f N\n', F_values(end));
    fprintf('Maximum mid-span deflection: %.3f m\n', deflection_midspan(end));
    fprintf('Maximum damage: %.4f\n', max_damage(end));
    fprintf('Mid-span curvature at final step: %.4f 1/m\n', curvature_midspan(end));
    
    % Debug: Check if curvatures are reasonable
    fprintf('Curvature range at final step: [%.2e, %.2e] 1/m\n', ...
            min(chi_history(:, final_step)), max(chi_history(:, final_step)));
    fprintf('Deflection range at final step: [%.2e, %.2e] m\n', ...
            min(deflection_history(:, final_step)), max(deflection_history(:, final_step)));
    
    if max(max_damage) > 0.95
        failure_force = F_values(find(max_damage > 0.95, 1));
        fprintf('Failure force: %.1f N\n', failure_force);
    end
end
