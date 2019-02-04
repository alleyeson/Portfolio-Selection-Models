#Regular Markowitz Algorithm 
Marko_weights <- function(x,targ_mu){
  library("MASS")
  p <- length(x[1,])
  ## with mean mu 
  mu <- (colMeans(x))
  ##and variance covariance 
  ret_var <- var(x)
  var_inverse <- ginv(ret_var) ##inverse of matrix
  inv_check <- t(var_inverse) %*% ret_var ##check if inverse is inverse
  
  ## More intimate parameters 
  B_temp <- t(var_inverse) %*% mu 
  b <- mu %*% B_temp
  ones <- rep(1,p)
  a<- ones%*%B_temp
  c <- ones%*% (t(var_inverse)%*%ones)
  d <- b*c - a*a
  #target returns be some value, let's use the one from the input of the function 
  target_returns <- targ_mu
  ####
  cd<- c/d
  ad <- a/d
  bd <- b/d
  muCoeff <- as.vector(cd*target_returns - ad)
  oneCoeff <- as.vector(bd - ad*target_returns)
  w <- var_inverse%*%(muCoeff * mu + oneCoeff*ones)
  w<- as.numeric(w)
  return(w)
}

## Sharpe Weights Function 
Port_port_Sharpe<-function(x,mat_A){
  mu <- colMeans(x)
  ret_var <- var(x)
  #mat_A<- cbind(w_0, w_o, w, w_2,w_sim)
  num_Mat <- ret_var%*%mat_A
  mat_A_T<- t(mat_A)
  num_Mat_num<- t(mat_A)%*%num_Mat
  inv_Mat_num<- ginv(num_Mat_num)
  ##inv_Mat_num%*%num_Mat_num
  
  mu<- as.numeric(mu)
  A_t_u <- t(mat_A)%*%mu
  one_A_t<- rep(1, length(A_t_u))
  diff_vect_au1<- A_t_u - one_A_t
  
  approx_j <- inv_Mat_num%*%diff_vect_au1
  
  scale_denum <- 1/t(one_A_t)%*%approx_j
  scale_denum<- as.numeric(scale_denum)
  j <- scale_denum*approx_j
  
  one_A_t%*%j
  
  w_recur<- mat_A%*%j ### THE TRUE j and combination of porfolios to be used 
  
  return(w_recur)
  
}