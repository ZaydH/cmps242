This program  is written in Matlab, but should be runnable in Octave (although I did not specify test it).  The main function's prototype is:
 
hw01(degree, lambdas, k, ignoreBias, offset_term);
 
To run it, I recommend you call something like:
 
lambdas = build_exponential_lambdas(2, 1024);
hw01(10, lambdas, 10, false);
 
If you want to enable the bias correction (i.e., ignore w_0), you would call:
 
hw01(10, lambdas, 10, true);
 
In addition, our work uses the Matlab "export_fig" add-on.  You can download it from the Mathworks File Exchange (https://www.mathworks.com/matlabcentral/fileexchange/23629-export-fig).  If you cannot or do not use want to install it, you can just comment out the "export_fig" calls, and it should work.  This is also unlikely to work to Octave.

To empirically observe the effect of excluding offset in regularization, you can pass to the "hw01" function an arbitrary value for "offset_term," which is added to the target value for both the test and training sets.  When "ignoreBias" is True, the algorithm should perform the same irrespective of the value of "offset_term".
