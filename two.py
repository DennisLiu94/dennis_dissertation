import qzliu_util as qzl

data=qzl.load_data('sen_dis.pkl')
total=sum(data)
print total
line=total*0.5
accu=0
for i in range(21):
    accu+=data[i]

print float(accu)/total
