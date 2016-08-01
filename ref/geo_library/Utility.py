class Utility(object):
    @staticmethod
    def ContainsList(listList, listItem):
        listItem.sort()
        for i in range(0, len(listList)):
            temp = listList[i]
            if(len(temp) == len(listItem)):
                temp.sort()
                itemEqual = True
                for j in range(0, len(listItem)):
                    if(temp[j] != listItem[j]):
                        itemEqual = False
                        break
                if(itemEqual):
                   return True
            else:
                return False
        return False