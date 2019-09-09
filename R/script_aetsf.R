library(tsensembler)

# list input files
fi = list.files("./st/")
Ks = c(96,24,7,288,288,4,12,96,24,4)

measures = matrix(0, nrow = 2, ncol = 10)

for(f in 1:10){print(fi[f])
  
  serie = read.csv2(paste("./st/",fi[f], sep = ""), stringsAsFactors = F, dec = ".")
  rg = sort(serie$n_day_, index.return = T)
  dim(serie)
  table(serie$time_)
  s = serie$val_[rg$ix]
  l = length(s)
  k = Ks[f]
  
  ns = l/k
  ntr = floor(ns*0.85)
  nte = ns - ntr
  
  ntr
  nte
  
  pred = matrix(0, nte-1, k)
  for(j in 1:(nrow(pred))){ print(j)
    ss = ts(s[1:(k*(ntr+j))], frequency = k)
    ser = xts(ss, date_decimal(index(ss)))
    train = embed_timeseries(ser,k)

    specs <- model_specs(learner = c("bm_glm","bm_svr","bm_randomforest", "bm_ffnn"), learner_pars = list(bm_glm = list(alpha = c(0, .5, 1)), bm_svr = list(kernel = c("rbfdot", "polydot"), C = c(1,3)), bm_ffnn = list(size = 10), bm_randomforest = list(num.trees = 50)))
    
    model <- ADE(target~., train, specs)
    p = forecast(model, h = k)
    pred[j,] = p
  }

  test = matrix(s[(k*(ntr+1)+1):length(s)], ncol = k, byrow = T)
  res = sapply(c(1:nrow(pred)), function(x){mean((pred[x, ]-as.numeric(test[x,]))^2)})
  
  measures[1,f] = mean(res)
  measures[2,f] = mae(pred, test)
  print(measures)
}