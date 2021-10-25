query="what is chicken pox"
query_list=query.split(' ')
find='chicken sox'
find_tuple=tuple(find.split(" "))
print(find_tuple)
lst=[]
for i in range(len(query_list)):
    lst.append((query_list[i]))
    lst.append((query_list[i-2],query_list[i-1],query_list[i]))
    lst.append((query_list[i-1],query_list[i]))
for i in range(len(lst)):
    if find_tuple==lst[i]:
        print(True)
        break
