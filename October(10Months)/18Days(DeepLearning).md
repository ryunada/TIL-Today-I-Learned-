
# Selenium


```python
# pip install selenium
```

## 경로 설정


```python
import os
print(os.getcwd())
```

    /Users/ryu/Desktop/데스크탑 - ryuseungho의 MacBook Air/2022/Bigdata/DeepLearning/Kim_Professor_10:11(2weeks)



```python
os.chdir('/Users/ryu')
print(os.getcwd())
```

    /Users/ryu



```python
from selenium import webdriver as wd
url = 'http://www.naver.com'

# 크롬 드라이브 로드
driver = wd.Chrome('./chromedriver')

# page laod
driver.get(url)
```

    /var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/ipykernel_4709/697203445.py:5: DeprecationWarning: executable_path has been deprecated, please pass in a Service object
      driver = wd.Chrome('./chromedriver')


## Naver 메인 페이지 검색

---
## 블로그 검색


```python
import os
import sys
import urllib.request

client_id = "eGaTUY3HVt4LzOMY6fWT"
client_secret = "klCa7Y07ml"

encText = urllib.parse.quote("강남역")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)

response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
```

    {
    	"lastBuildDate":"Tue, 18 Oct 2022 13:09:47 +0900",
    	"total":1948615,
    	"start":1,
    	"display":10,
    	"items":[
    		{
    			"title":"고퀄리티였던 <b>강남역<\/b> 고기집",
    			"link":"https:\/\/blog.naver.com\/shalom_2\/222887747770",
    			"description":"도마3 서울 서초구 서초대로78길 48 송림빌딩 1층 101호 02-525-2159 영업시간 : 11:30 - 23:00 지난 휴일에 친구들과 저녁 먹으러 강남에서 모여 지인이 극찬했던 <b>강남역<\/b> 고기집을 방문했어요. 고소한 마블링을 자랑하는... ",
    			"bloggername":"Hello, Rosie :)",
    			"bloggerlink":"blog.naver.com\/shalom_2",
    			"postdate":"20220929"
    		},
    		{
    			"title":"소장하고싶은 <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/01o01o01o\/222869413501",
    			"description":"은은한 향의 와인이 입속에 감미롭게 어려와서 마지막까지 흐뭇했던 <b>강남역<\/b> 삼겹살이에요. 주소 : 강남구 역삼동 619-14 1층 봉우화로 번호 : 02-558-8452 영업시간 : 11:30 ~ 23:00 대표메뉴 : 생삼겹살 12,000원... ",
    			"bloggername":"THE G.U.",
    			"bloggerlink":"blog.naver.com\/01o01o01o",
    			"postdate":"20220907"
    		},
    		{
    			"title":"제대로였던 <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/duvmfh0327\/222851553244",
    			"description":"그렇게 국수까지 배가 부르다고 이야기 하면서도 자꾸만 손이가게 되는 <b>강남역<\/b> 삼겹살이었어요. 다음번에도 기회가 된다면 다시 찾아올 곳이라며 대 만족했던 곳이랍니다. 주소:서울 강남구 강남대로98길12 1층... ",
    			"bloggername":"깡다맘",
    			"bloggerlink":"blog.naver.com\/duvmfh0327",
    			"postdate":"20220818"
    		},
    		{
    			"title":"[봉우화로] <b>강남역<\/b> 고기집 \/ ft. <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/ejh03107\/222886614701",
    			"description":"주소: 강남구 역삼동 619-14 1층 봉우화로 번호: 02-558-8452 영업시간: 11:30 - 23:00 이번에 방문했던 식당은 <b>강남역<\/b> 11번 출구에서 도보로 5분이면 가는 거리이더라고요 상가 건물에 하얀 간판이 보여서 바로 찾을 수... ",
    			"bloggername":"☁️ 둥글게 살자 :) ☁️",
    			"bloggerlink":"blog.naver.com\/ejh03107",
    			"postdate":"20220929"
    		},
    		{
    			"title":"환상적인 <b>강남역<\/b> 고기집",
    			"link":"https:\/\/blog.naver.com\/benison7\/222881772812",
    			"description":"업체명 : 유니네 위치안내 : 서울 강남구 강남대로98길12 1층 전화번호 : 02-562-1806 영업시간 : 12:00~23:00 저희가 방문한 매장은 <b>강남역<\/b> 11번 출구에서 도보로 5분이면 도착했어요. 건물 외관이 고급스러운데다가... ",
    			"bloggername":"마시멜로의 소소한 일상♡",
    			"bloggerlink":"blog.naver.com\/benison7",
    			"postdate":"20220922"
    		},
    		{
    			"title":"<b>강남역<\/b>pt 전문적이고 균형잡힌 관리",
    			"link":"https:\/\/blog.naver.com\/exp8824\/222899420399",
    			"description":"<b>강남역<\/b> pt 생각 있으시다면 MN강남점 괜찮다고 말씀드리고 싶네요. 운영시간 넉넉한데다 기구 관리도 잘 되고, 위생적이라 이런 거 민감하신 분들에게도 자신있게 추천합니다.",
    			"bloggername":"대부시리 사냥꾼, 마다이의 낚시이야기",
    			"bloggerlink":"blog.naver.com\/exp8824",
    			"postdate":"20221013"
    		},
    		{
    			"title":"끝내줬던 <b>강남역<\/b> 소고기",
    			"link":"https:\/\/blog.naver.com\/soojini65\/222880819631",
    			"description":"주소 : 서울 서초구 서초대로78길 48 송림빌딩 1층 101호 도마3 전화번호 : 02-525-2159 영업시간 : 11:30 - 23:00 이날 찾아간 도마3은 <b>강남역<\/b> 5번 출구로 나와 걸으니 3분 정도 소요되었어요. 역 근처라 술한잔을 하기 위해... ",
    			"bloggername":"Hurricane 준선아빠의 물생활&맛집&투자일지etc",
    			"bloggerlink":"blog.naver.com\/soojini65",
    			"postdate":"20220921"
    		},
    		{
    			"title":"<b>강남역<\/b>필라테스 올바른 자세찾기",
    			"link":"https:\/\/blog.naver.com\/gbh0828\/222893526954",
    			"description":"하지만 최근에는 통증도 생기고 체중이 급격하게 늘면서 하중도 실리는 거 같길래 뭐라도 시작하자는 생각으로 여러 개를 비교하다가 <b>강남역<\/b> 1분거리에 괜찮은 <b>강남역<\/b>필라테스 전문점이 있어서 등록하고 다니게... ",
    			"bloggername":"무슨 게임을 할까",
    			"bloggerlink":"blog.naver.com\/gbh0828",
    			"postdate":"20221006"
    		},
    		{
    			"title":"품격있던 <b>강남역<\/b> 레스토랑",
    			"link":"https:\/\/blog.naver.com\/lymlove\/222861372267",
    			"description":"서울 강남구 테헤란로5길 24 번호 : 02-538-5067 운영시간 : 11:30 - 22:00 \/ 라스트오더 21:00 주차여부 : 전용주차장 보유 대표메뉴 : 마르게리따 18,000원 이날 향했던 레스토랑은 <b>강남역<\/b> 11번 출구에서 290미터만 이동하면... ",
    			"bloggername":"예쁜시간",
    			"bloggerlink":"blog.naver.com\/lymlove",
    			"postdate":"20220829"
    		},
    		{
    			"title":"<b>강남역<\/b> 룸식당 육목원 아늑한 분위기가 매력적인 곳",
    			"link":"https:\/\/blog.naver.com\/hcs4295\/222899467246",
    			"description":"육목원은 <b>강남역<\/b> 5번 출구에서 도보로 3분 소요되는 거리에 있었는데요. 한우부터 돼지갈비까지... 많은 분들이 <b>강남역<\/b> 룸식당에 오시면 이것부터 드시는 것 같았거든요. 주문을 시킨 후 앉아서 잠시 기다리니... ",
    			"bloggername":"라이마스튜디오",
    			"bloggerlink":"blog.naver.com\/hcs4295",
    			"postdate":"20221013"
    		}
    	]
    }


[파라미터](https://developers.naver.com/docs/serviceapi/search/blog/blog.md#%ED%8C%8C%EB%9D%BC%EB%AF%B8%ED%84%B0)

```python
url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&display=20&start=1&sort=sim"
url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&display=20&start=1&sort=date"
```


```python
import os
import sys
import urllib.request

client_id = "eGaTUY3HVt4LzOMY6fWT"
client_secret = "klCa7Y07ml"

encTexte= urllib.parse.quote("강남역")
url = "https://openapi.naver.com/v1/search/blog?query=" + encText + "&display=20&start=1&sort=sim" # JSON 결과

#url = "https://openapi.naver.com/v1/search/blog?query=강남역&display=20&start=50&sort=sim" 
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)

response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
    
type(eval(response_body.decode('utf-8')))
rData = eval(response_body.decode('utf-8'))
```

    {
    	"lastBuildDate":"Tue, 18 Oct 2022 13:11:40 +0900",
    	"total":1948615,
    	"start":1,
    	"display":20,
    	"items":[
    		{
    			"title":"고퀄리티였던 <b>강남역<\/b> 고기집",
    			"link":"https:\/\/blog.naver.com\/shalom_2\/222887747770",
    			"description":"도마3 서울 서초구 서초대로78길 48 송림빌딩 1층 101호 02-525-2159 영업시간 : 11:30 - 23:00 지난 휴일에 친구들과 저녁 먹으러 강남에서 모여 지인이 극찬했던 <b>강남역<\/b> 고기집을 방문했어요. 고소한 마블링을 자랑하는... ",
    			"bloggername":"Hello, Rosie :)",
    			"bloggerlink":"blog.naver.com\/shalom_2",
    			"postdate":"20220929"
    		},
    		{
    			"title":"소장하고싶은 <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/01o01o01o\/222869413501",
    			"description":"은은한 향의 와인이 입속에 감미롭게 어려와서 마지막까지 흐뭇했던 <b>강남역<\/b> 삼겹살이에요. 주소 : 강남구 역삼동 619-14 1층 봉우화로 번호 : 02-558-8452 영업시간 : 11:30 ~ 23:00 대표메뉴 : 생삼겹살 12,000원... ",
    			"bloggername":"THE G.U.",
    			"bloggerlink":"blog.naver.com\/01o01o01o",
    			"postdate":"20220907"
    		},
    		{
    			"title":"제대로였던 <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/duvmfh0327\/222851553244",
    			"description":"그렇게 국수까지 배가 부르다고 이야기 하면서도 자꾸만 손이가게 되는 <b>강남역<\/b> 삼겹살이었어요. 다음번에도 기회가 된다면 다시 찾아올 곳이라며 대 만족했던 곳이랍니다. 주소:서울 강남구 강남대로98길12 1층... ",
    			"bloggername":"깡다맘",
    			"bloggerlink":"blog.naver.com\/duvmfh0327",
    			"postdate":"20220818"
    		},
    		{
    			"title":"[봉우화로] <b>강남역<\/b> 고기집 \/ ft. <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/ejh03107\/222886614701",
    			"description":"주소: 강남구 역삼동 619-14 1층 봉우화로 번호: 02-558-8452 영업시간: 11:30 - 23:00 이번에 방문했던 식당은 <b>강남역<\/b> 11번 출구에서 도보로 5분이면 가는 거리이더라고요 상가 건물에 하얀 간판이 보여서 바로 찾을 수... ",
    			"bloggername":"☁️ 둥글게 살자 :) ☁️",
    			"bloggerlink":"blog.naver.com\/ejh03107",
    			"postdate":"20220929"
    		},
    		{
    			"title":"환상적인 <b>강남역<\/b> 고기집",
    			"link":"https:\/\/blog.naver.com\/benison7\/222881772812",
    			"description":"업체명 : 유니네 위치안내 : 서울 강남구 강남대로98길12 1층 전화번호 : 02-562-1806 영업시간 : 12:00~23:00 저희가 방문한 매장은 <b>강남역<\/b> 11번 출구에서 도보로 5분이면 도착했어요. 건물 외관이 고급스러운데다가... ",
    			"bloggername":"마시멜로의 소소한 일상♡",
    			"bloggerlink":"blog.naver.com\/benison7",
    			"postdate":"20220922"
    		},
    		{
    			"title":"<b>강남역<\/b>pt 전문적이고 균형잡힌 관리",
    			"link":"https:\/\/blog.naver.com\/exp8824\/222899420399",
    			"description":"<b>강남역<\/b> pt 생각 있으시다면 MN강남점 괜찮다고 말씀드리고 싶네요. 운영시간 넉넉한데다 기구 관리도 잘 되고, 위생적이라 이런 거 민감하신 분들에게도 자신있게 추천합니다.",
    			"bloggername":"대부시리 사냥꾼, 마다이의 낚시이야기",
    			"bloggerlink":"blog.naver.com\/exp8824",
    			"postdate":"20221013"
    		},
    		{
    			"title":"끝내줬던 <b>강남역<\/b> 소고기",
    			"link":"https:\/\/blog.naver.com\/soojini65\/222880819631",
    			"description":"주소 : 서울 서초구 서초대로78길 48 송림빌딩 1층 101호 도마3 전화번호 : 02-525-2159 영업시간 : 11:30 - 23:00 이날 찾아간 도마3은 <b>강남역<\/b> 5번 출구로 나와 걸으니 3분 정도 소요되었어요. 역 근처라 술한잔을 하기 위해... ",
    			"bloggername":"Hurricane 준선아빠의 물생활&맛집&투자일지etc",
    			"bloggerlink":"blog.naver.com\/soojini65",
    			"postdate":"20220921"
    		},
    		{
    			"title":"<b>강남역<\/b>필라테스 올바른 자세찾기",
    			"link":"https:\/\/blog.naver.com\/gbh0828\/222893526954",
    			"description":"하지만 최근에는 통증도 생기고 체중이 급격하게 늘면서 하중도 실리는 거 같길래 뭐라도 시작하자는 생각으로 여러 개를 비교하다가 <b>강남역<\/b> 1분거리에 괜찮은 <b>강남역<\/b>필라테스 전문점이 있어서 등록하고 다니게... ",
    			"bloggername":"무슨 게임을 할까",
    			"bloggerlink":"blog.naver.com\/gbh0828",
    			"postdate":"20221006"
    		},
    		{
    			"title":"품격있던 <b>강남역<\/b> 레스토랑",
    			"link":"https:\/\/blog.naver.com\/lymlove\/222861372267",
    			"description":"서울 강남구 테헤란로5길 24 번호 : 02-538-5067 운영시간 : 11:30 - 22:00 \/ 라스트오더 21:00 주차여부 : 전용주차장 보유 대표메뉴 : 마르게리따 18,000원 이날 향했던 레스토랑은 <b>강남역<\/b> 11번 출구에서 290미터만 이동하면... ",
    			"bloggername":"예쁜시간",
    			"bloggerlink":"blog.naver.com\/lymlove",
    			"postdate":"20220829"
    		},
    		{
    			"title":"<b>강남역<\/b> 룸식당 육목원 아늑한 분위기가 매력적인 곳",
    			"link":"https:\/\/blog.naver.com\/hcs4295\/222899467246",
    			"description":"육목원은 <b>강남역<\/b> 5번 출구에서 도보로 3분 소요되는 거리에 있었는데요. 한우부터 돼지갈비까지... 많은 분들이 <b>강남역<\/b> 룸식당에 오시면 이것부터 드시는 것 같았거든요. 주문을 시킨 후 앉아서 잠시 기다리니... ",
    			"bloggername":"라이마스튜디오",
    			"bloggerlink":"blog.naver.com\/hcs4295",
    			"postdate":"20221013"
    		},
    		{
    			"title":"향긋했던 <b>강남역<\/b> 커피",
    			"link":"https:\/\/blog.naver.com\/s2hkbm\/222893473386",
    			"description":"<b>강남역<\/b> 원퍼밀 주소 : 서울 강남구 강남대로106길 7 1층 영업시간 : 금 08:00 - 22:10, 22:00 라스트오더 전화번호 : 070-8844-1591 메뉴 : 아메리카노 3,000원, 카야버터토스트 5,900원 그날 찾아간 원퍼밀은 신논현역... ",
    			"bloggername":"그리는 보미",
    			"bloggerlink":"blog.naver.com\/s2hkbm",
    			"postdate":"20221006"
    		},
    		{
    			"title":"심쿵했던 <b>강남역<\/b> 피자",
    			"link":"https:\/\/blog.naver.com\/dolice62\/222886839494",
    			"description":"서울 강남구 테헤란로5길 24 번호 : 02-538-5067 운영시간 : 11:30 - 22:00 \/ 라스트오더 21:00 주차여부 : 전용주차장 보유 대표메뉴 : 프리마베라 24,000원 이번에 알게 된 피자집은 <b>강남역<\/b> 11번 출구에서 차로 292미터만... ",
    			"bloggername":"도리스블로그",
    			"bloggerlink":"blog.naver.com\/dolice62",
    			"postdate":"20220928"
    		},
    		{
    			"title":"수준 높았던 <b>강남역<\/b> 파스타",
    			"link":"https:\/\/blog.naver.com\/movie457\/222861358531",
    			"description":"입가심은 자몽에이드로 했는데요. 톡톡 튀는 탄산이 막힌 속을 한방에 뚫어주어 마지막까지도 퍼펙트했던 <b>강남역<\/b> 파스타였답니다~ 주소 : 서울 강남구 테헤란로5길 24 번호 : 02-538-5067 시간 : 매일 11:30-22:00",
    			"bloggername":"생각과 꿈의 크기만큼 자랐어요",
    			"bloggerlink":"blog.naver.com\/movie457",
    			"postdate":"20220829"
    		},
    		{
    			"title":"<b>강남역<\/b> pt 건강하고 체계적인 관리",
    			"link":"https:\/\/blog.naver.com\/spapabobo\/222881467491",
    			"description":"벌써부터 건강해진다는 느낌도 들면서 삶의 질도 높아지게 만들어준 <b>강남역<\/b> pt 는 동료나 지인 외에도 가족에게까지 추천하고 싶은 마음이 들었습니다. 마지막으로 이곳 강남 최고의 프리미엄 태닝시설을... ",
    			"bloggername":"haejihn",
    			"bloggerlink":"blog.naver.com\/spapabobo",
    			"postdate":"20220922"
    		},
    		{
    			"title":"마음에 들었던 <b>강남역<\/b> 회식",
    			"link":"https:\/\/blog.naver.com\/fun0218\/222902715114",
    			"description":"달짝지근한 과육이 텁텁해진 입안을 말끔하게 헹궈주어 입가심까지 완벽했던 <b>강남역<\/b> 회식이에요~ 주소 : 서울 강남구 테헤란로5길 24 1층 번호 : 02-538-5067 운영시간 : 매일 11:30 - 22:00 주차여부 : O 대표메뉴... ",
    			"bloggername":"찰칵",
    			"bloggerlink":"blog.naver.com\/fun0218",
    			"postdate":"20221017"
    		},
    		{
    			"title":"우월하던 <b>강남역<\/b> 고기집",
    			"link":"https:\/\/blog.naver.com\/dasul119\/222875902008",
    			"description":"얼마 전 친구들과 강남 쪽에 놀러 갔다가 입소문이 자자한 <b>강남역<\/b> 고기집에 들러 식사를 해봤는데요. 30년 노하우가 담긴 돼지갈비와 여러 고기를 숯불에다가 노릇하게 구워 맛보고 와서 찾아간 보람이 느껴진... ",
    			"bloggername":".",
    			"bloggerlink":"blog.naver.com\/dasul119",
    			"postdate":"20220916"
    		},
    		{
    			"title":"완전 반한 <b>강남역<\/b> 고기집",
    			"link":"https:\/\/blog.naver.com\/yskim243\/222847532452",
    			"description":"달달하면서도 강한 내음이 입을 씻어주니 시작부터 끝까지 부족함이 없었던 <b>강남역<\/b> 고기집이랍니다. 위치 : 서울 강남구 봉은사로 18길 76 스타 팰리스 전화번호 : 02-558-8452 영업시간 : 11:30 ~ 23:30",
    			"bloggername":"김이든의 먹고 삽시다.",
    			"bloggerlink":"blog.naver.com\/yskim243",
    			"postdate":"20220813"
    		},
    		{
    			"title":"퀄리티가 남다른 <b>강남역<\/b> 파스타 | 블랙스테이크",
    			"link":"https:\/\/blog.naver.com\/sia855\/222887466006",
    			"description":"마침 강남에 볼일도 있고 근처 사는 친구 불러서 <b>강남역<\/b> 파스타로도 유명한 블랙 스테이크에서 식사를 하고 왔어요. 부담 없는 가격이었는데 맛과 퀄리티가 좋아 공유해 드려 볼게요! 블랙스테이크 강남점 메뉴... ",
    			"bloggername":"행복한 시야의 블로그",
    			"bloggerlink":"blog.naver.com\/sia855",
    			"postdate":"20221001"
    		},
    		{
    			"title":"<b>강남역<\/b> 소고기, 역삼역 고기집, 강남 회식장소 진씨화로",
    			"link":"https:\/\/blog.naver.com\/minheee12\/222896814081",
    			"description":"진씨화로 <b>강남역<\/b> 소고기 맛집 <b>강남역<\/b> 회식장소 - 가격 : 한우등심 56,000원 - 주차 : 가능 - 영업 : 11:00~22:00 - 예약 : 네이버예약 https:\/\/naver.me\/G7KZNhkp <b>강남역<\/b>에는 워낙 많은 고기집이 있는데요! <b>강남역<\/b> 소고기, 한우... ",
    			"bloggername":"미닝미니닝의 일상",
    			"bloggerlink":"blog.naver.com\/minheee12",
    			"postdate":"20221010"
    		},
    		{
    			"title":"역대급이던 <b>강남역<\/b> 삼겹살",
    			"link":"https:\/\/blog.naver.com\/blewsky\/222857186535",
    			"description":"올라와서 <b>강남역<\/b> 인근에서 만나 하루 놀기로 했어요. 맛있는 걸 사주고 싶어 고민하다가 지인이 적극적으로 추천해준 곳이 있어서 망설임 없이 다녀와봤답니다. 고기를 좋아하는 친구를 위해서 <b>강남역<\/b>... ",
    			"bloggername":"푸른고니(blewsky)",
    			"bloggerlink":"blog.naver.com\/blewsky",
    			"postdate":"20220825"
    		}
    	]
    }



```python
len( rData['items'] )
```




    20



순신한 데이터의 내용물

- lastBuildDate
- total
- start
- display
- items

###  다른 코드로 접속


```python
import requests

url = "https://openapi.naver.com/v1/search/blog.json"

#url = "https://openapi.naver.com/v1/search/blog?query=강남역&display=20&start=50&sort=sim" 
#request = urllib.request.Request(url)
#request.add_header("X-Naver-Client-Id",client_id)
#request.add_header("X-Naver-Client-Secret",client_secret)
headers =  {
    'X-Naver-Client-Id': 'eGaTUY3HVt4LzOMY6fWT',
    'X-Naver-Client-Secret': 'klCa7Y07ml'
}

payload = {
    'query': '강남역',
    'display': '20',   # 20개
    'start':1,
    'sort':'sim'       # 정렬은 유사한것으로
}

response = requests.get(url,headers=headers, params=payload)

```


```python
dic_res = response.json()

for key in dic_res:
    print( key,':',dic_res[key] )
```

    lastBuildDate : Tue, 18 Oct 2022 13:12:46 +0900
    total : 1948612
    start : 1
    display : 20
    items : [{'title': '고퀄리티였던 <b>강남역</b> 고기집', 'link': 'https://blog.naver.com/shalom_2/222887747770', 'description': '도마3 서울 서초구 서초대로78길 48 송림빌딩 1층 101호 02-525-2159 영업시간 : 11:30 - 23:00 지난 휴일에 친구들과 저녁 먹으러 강남에서 모여 지인이 극찬했던 <b>강남역</b> 고기집을 방문했어요. 고소한 마블링을 자랑하는... ', 'bloggername': 'Hello, Rosie :)', 'bloggerlink': 'blog.naver.com/shalom_2', 'postdate': '20220929'}, {'title': '소장하고싶은 <b>강남역</b> 삼겹살', 'link': 'https://blog.naver.com/01o01o01o/222869413501', 'description': '은은한 향의 와인이 입속에 감미롭게 어려와서 마지막까지 흐뭇했던 <b>강남역</b> 삼겹살이에요. 주소 : 강남구 역삼동 619-14 1층 봉우화로 번호 : 02-558-8452 영업시간 : 11:30 ~ 23:00 대표메뉴 : 생삼겹살 12,000원... ', 'bloggername': 'THE G.U.', 'bloggerlink': 'blog.naver.com/01o01o01o', 'postdate': '20220907'}, {'title': '제대로였던 <b>강남역</b> 삼겹살', 'link': 'https://blog.naver.com/duvmfh0327/222851553244', 'description': '그렇게 국수까지 배가 부르다고 이야기 하면서도 자꾸만 손이가게 되는 <b>강남역</b> 삼겹살이었어요. 다음번에도 기회가 된다면 다시 찾아올 곳이라며 대 만족했던 곳이랍니다. 주소:서울 강남구 강남대로98길12 1층... ', 'bloggername': '깡다맘', 'bloggerlink': 'blog.naver.com/duvmfh0327', 'postdate': '20220818'}, {'title': '[봉우화로] <b>강남역</b> 고기집 / ft. <b>강남역</b> 삼겹살', 'link': 'https://blog.naver.com/ejh03107/222886614701', 'description': '주소: 강남구 역삼동 619-14 1층 봉우화로 번호: 02-558-8452 영업시간: 11:30 - 23:00 이번에 방문했던 식당은 <b>강남역</b> 11번 출구에서 도보로 5분이면 가는 거리이더라고요 상가 건물에 하얀 간판이 보여서 바로 찾을 수... ', 'bloggername': '☁️ 둥글게 살자 :) ☁️', 'bloggerlink': 'blog.naver.com/ejh03107', 'postdate': '20220929'}, {'title': '환상적인 <b>강남역</b> 고기집', 'link': 'https://blog.naver.com/benison7/222881772812', 'description': '업체명 : 유니네 위치안내 : 서울 강남구 강남대로98길12 1층 전화번호 : 02-562-1806 영업시간 : 12:00~23:00 저희가 방문한 매장은 <b>강남역</b> 11번 출구에서 도보로 5분이면 도착했어요. 건물 외관이 고급스러운데다가... ', 'bloggername': '마시멜로의 소소한 일상♡', 'bloggerlink': 'blog.naver.com/benison7', 'postdate': '20220922'}, {'title': '<b>강남역</b>pt 전문적이고 균형잡힌 관리', 'link': 'https://blog.naver.com/exp8824/222899420399', 'description': '<b>강남역</b> pt 생각 있으시다면 MN강남점 괜찮다고 말씀드리고 싶네요. 운영시간 넉넉한데다 기구 관리도 잘 되고, 위생적이라 이런 거 민감하신 분들에게도 자신있게 추천합니다.', 'bloggername': '대부시리 사냥꾼, 마다이의 낚시이야기', 'bloggerlink': 'blog.naver.com/exp8824', 'postdate': '20221013'}, {'title': '끝내줬던 <b>강남역</b> 소고기', 'link': 'https://blog.naver.com/soojini65/222880819631', 'description': '주소 : 서울 서초구 서초대로78길 48 송림빌딩 1층 101호 도마3 전화번호 : 02-525-2159 영업시간 : 11:30 - 23:00 이날 찾아간 도마3은 <b>강남역</b> 5번 출구로 나와 걸으니 3분 정도 소요되었어요. 역 근처라 술한잔을 하기 위해... ', 'bloggername': 'Hurricane 준선아빠의 물생활&맛집&투자일지etc', 'bloggerlink': 'blog.naver.com/soojini65', 'postdate': '20220921'}, {'title': '<b>강남역</b>필라테스 올바른 자세찾기', 'link': 'https://blog.naver.com/gbh0828/222893526954', 'description': '하지만 최근에는 통증도 생기고 체중이 급격하게 늘면서 하중도 실리는 거 같길래 뭐라도 시작하자는 생각으로 여러 개를 비교하다가 <b>강남역</b> 1분거리에 괜찮은 <b>강남역</b>필라테스 전문점이 있어서 등록하고 다니게... ', 'bloggername': '무슨 게임을 할까', 'bloggerlink': 'blog.naver.com/gbh0828', 'postdate': '20221006'}, {'title': '품격있던 <b>강남역</b> 레스토랑', 'link': 'https://blog.naver.com/lymlove/222861372267', 'description': '서울 강남구 테헤란로5길 24 번호 : 02-538-5067 운영시간 : 11:30 - 22:00 / 라스트오더 21:00 주차여부 : 전용주차장 보유 대표메뉴 : 마르게리따 18,000원 이날 향했던 레스토랑은 <b>강남역</b> 11번 출구에서 290미터만 이동하면... ', 'bloggername': '예쁜시간', 'bloggerlink': 'blog.naver.com/lymlove', 'postdate': '20220829'}, {'title': '<b>강남역</b> 룸식당 육목원 아늑한 분위기가 매력적인 곳', 'link': 'https://blog.naver.com/hcs4295/222899467246', 'description': '육목원은 <b>강남역</b> 5번 출구에서 도보로 3분 소요되는 거리에 있었는데요. 한우부터 돼지갈비까지... 많은 분들이 <b>강남역</b> 룸식당에 오시면 이것부터 드시는 것 같았거든요. 주문을 시킨 후 앉아서 잠시 기다리니... ', 'bloggername': '라이마스튜디오', 'bloggerlink': 'blog.naver.com/hcs4295', 'postdate': '20221013'}, {'title': '향긋했던 <b>강남역</b> 커피', 'link': 'https://blog.naver.com/s2hkbm/222893473386', 'description': '<b>강남역</b> 원퍼밀 주소 : 서울 강남구 강남대로106길 7 1층 영업시간 : 금 08:00 - 22:10, 22:00 라스트오더 전화번호 : 070-8844-1591 메뉴 : 아메리카노 3,000원, 카야버터토스트 5,900원 그날 찾아간 원퍼밀은 신논현역... ', 'bloggername': '그리는 보미', 'bloggerlink': 'blog.naver.com/s2hkbm', 'postdate': '20221006'}, {'title': '심쿵했던 <b>강남역</b> 피자', 'link': 'https://blog.naver.com/dolice62/222886839494', 'description': '서울 강남구 테헤란로5길 24 번호 : 02-538-5067 운영시간 : 11:30 - 22:00 / 라스트오더 21:00 주차여부 : 전용주차장 보유 대표메뉴 : 프리마베라 24,000원 이번에 알게 된 피자집은 <b>강남역</b> 11번 출구에서 차로 292미터만... ', 'bloggername': '도리스블로그', 'bloggerlink': 'blog.naver.com/dolice62', 'postdate': '20220928'}, {'title': '수준 높았던 <b>강남역</b> 파스타', 'link': 'https://blog.naver.com/movie457/222861358531', 'description': '입가심은 자몽에이드로 했는데요. 톡톡 튀는 탄산이 막힌 속을 한방에 뚫어주어 마지막까지도 퍼펙트했던 <b>강남역</b> 파스타였답니다~ 주소 : 서울 강남구 테헤란로5길 24 번호 : 02-538-5067 시간 : 매일 11:30-22:00', 'bloggername': '생각과 꿈의 크기만큼 자랐어요', 'bloggerlink': 'blog.naver.com/movie457', 'postdate': '20220829'}, {'title': '<b>강남역</b> pt 건강하고 체계적인 관리', 'link': 'https://blog.naver.com/spapabobo/222881467491', 'description': '벌써부터 건강해진다는 느낌도 들면서 삶의 질도 높아지게 만들어준 <b>강남역</b> pt 는 동료나 지인 외에도 가족에게까지 추천하고 싶은 마음이 들었습니다. 마지막으로 이곳 강남 최고의 프리미엄 태닝시설을... ', 'bloggername': 'haejihn', 'bloggerlink': 'blog.naver.com/spapabobo', 'postdate': '20220922'}, {'title': '마음에 들었던 <b>강남역</b> 회식', 'link': 'https://blog.naver.com/fun0218/222902715114', 'description': '달짝지근한 과육이 텁텁해진 입안을 말끔하게 헹궈주어 입가심까지 완벽했던 <b>강남역</b> 회식이에요~ 주소 : 서울 강남구 테헤란로5길 24 1층 번호 : 02-538-5067 운영시간 : 매일 11:30 - 22:00 주차여부 : O 대표메뉴... ', 'bloggername': '찰칵', 'bloggerlink': 'blog.naver.com/fun0218', 'postdate': '20221017'}, {'title': '우월하던 <b>강남역</b> 고기집', 'link': 'https://blog.naver.com/dasul119/222875902008', 'description': '얼마 전 친구들과 강남 쪽에 놀러 갔다가 입소문이 자자한 <b>강남역</b> 고기집에 들러 식사를 해봤는데요. 30년 노하우가 담긴 돼지갈비와 여러 고기를 숯불에다가 노릇하게 구워 맛보고 와서 찾아간 보람이 느껴진... ', 'bloggername': '.', 'bloggerlink': 'blog.naver.com/dasul119', 'postdate': '20220916'}, {'title': '완전 반한 <b>강남역</b> 고기집', 'link': 'https://blog.naver.com/yskim243/222847532452', 'description': '달달하면서도 강한 내음이 입을 씻어주니 시작부터 끝까지 부족함이 없었던 <b>강남역</b> 고기집이랍니다. 위치 : 서울 강남구 봉은사로 18길 76 스타 팰리스 전화번호 : 02-558-8452 영업시간 : 11:30 ~ 23:30', 'bloggername': '김이든의 먹고 삽시다.', 'bloggerlink': 'blog.naver.com/yskim243', 'postdate': '20220813'}, {'title': '퀄리티가 남다른 <b>강남역</b> 파스타 | 블랙스테이크', 'link': 'https://blog.naver.com/sia855/222887466006', 'description': '마침 강남에 볼일도 있고 근처 사는 친구 불러서 <b>강남역</b> 파스타로도 유명한 블랙 스테이크에서 식사를 하고 왔어요. 부담 없는 가격이었는데 맛과 퀄리티가 좋아 공유해 드려 볼게요! 블랙스테이크 강남점 메뉴... ', 'bloggername': '행복한 시야의 블로그', 'bloggerlink': 'blog.naver.com/sia855', 'postdate': '20221001'}, {'title': '<b>강남역</b> 소고기, 역삼역 고기집, 강남 회식장소 진씨화로', 'link': 'https://blog.naver.com/minheee12/222896814081', 'description': '진씨화로 <b>강남역</b> 소고기 맛집 <b>강남역</b> 회식장소 - 가격 : 한우등심 56,000원 - 주차 : 가능 - 영업 : 11:00~22:00 - 예약 : 네이버예약 https://naver.me/G7KZNhkp <b>강남역</b>에는 워낙 많은 고기집이 있는데요! <b>강남역</b> 소고기, 한우... ', 'bloggername': '미닝미니닝의 일상', 'bloggerlink': 'blog.naver.com/minheee12', 'postdate': '20221010'}, {'title': '역대급이던 <b>강남역</b> 삼겹살', 'link': 'https://blog.naver.com/blewsky/222857186535', 'description': '올라와서 <b>강남역</b> 인근에서 만나 하루 놀기로 했어요. 맛있는 걸 사주고 싶어 고민하다가 지인이 적극적으로 추천해준 곳이 있어서 망설임 없이 다녀와봤답니다. 고기를 좋아하는 친구를 위해서 <b>강남역</b>... ', 'bloggername': '푸른고니(blewsky)', 'bloggerlink': 'blog.naver.com/blewsky', 'postdate': '20220825'}]


## [ 실습 과제 ]
과제 1 - 검색 키워드 '인공지능'으로 블로그 검색을 하는데, 최신 글 순으로 처음부터 40개를 뽑아서 해당 블로그의 링크를 추출하라.

# 뉴스 검색
[뉴스 검색에 대한 설명](https://developers.naver.com/docs/serviceapi/search/news/news.md#%EB%89%B4%EC%8A%A4)

요청 URL
```python
https://openapi.naver.com/v1/search/news.json
```


```python
import requests

url = "https://openapi.naver.com/v1/search/news.json"

headers = {
    'X-Naver-Client-Id': 'eGaTUY3HVt4LzOMY6fWT',
    'X-Naver-Client-Secret': 'klCa7Y07ml'
}

payload = {
    'query' :  '안동역', # 검색어
    'display' : '20',  # 검색 갯수
    'start' : 1,       # 시작 번호
    'sort' : 'date'    # 정렬 방법
}

response = requests.get(url, headers = headers, params = payload)
```


```python
dic_res = response.json()

for key in dic_res:
    print(key, ':', dic_res[key])
```

    lastBuildDate : Tue, 18 Oct 2022 13:24:58 +0900
    total : 9979
    start : 1
    display : 20
    items : [{'title': '&apos;가요무대&apos; 가수 설운도·박상철·진성, &quot;항구의 남자, 삼바의 여인&quot; 히트곡 ...', 'originallink': 'http://www.topstarnews.net/news/articleView.html?idxno=14759903', 'link': 'http://www.topstarnews.net/news/articleView.html?idxno=14759903', 'description': '설운도와 진성은 공연 막바지에 또 무대에 올라 각각 &apos;보랏빛 엽서&apos;와 &apos;<b>안동역</b>에서&apos;를 부르며 진한 감동을 전하기도 했다. KBS1 중장년층 대상 음악 프로그램 &apos;가요무대&apos;는 매주 월요일 오후 10시에 방송된다.... ', 'pubDate': 'Mon, 17 Oct 2022 22:06:00 +0900'}, {'title': '&apos;가요무대&apos; 오늘(17일) 출연진 공개…송가인 박서진 현숙 박상철 설운도 등', 'originallink': 'http://sports.hankooki.com/news/articleView.html?idxno=6810002', 'link': 'http://sports.hankooki.com/news/articleView.html?idxno=6810002', 'description': '짝사랑(고복수) / 양지원 16. 항구의 남자(1)+무조건(1절+후렴반복) / 박상철 17. 요즘여자 요즘남자(1)+오빠는 잘 있단다 / 현숙 18. 보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자', 'pubDate': 'Mon, 17 Oct 2022 21:52:00 +0900'}, {'title': '[가요무대 출연진] 가수 설운도·김용임·진성·금잔디·신유·송가인·박상철...', 'originallink': 'https://www.sisamagazine.co.kr/news/articleView.html?idxno=469943', 'link': 'https://www.sisamagazine.co.kr/news/articleView.html?idxno=469943', 'description': '보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자 한편, 가요무대는 매 방송마다 출연진 나이부터 프로필까지 시청자들의 관심을 끌고 있다.', 'pubDate': 'Mon, 17 Oct 2022 21:40:00 +0900'}, {'title': '[가요무대 출연진] 가수 설운도·김용임·진성·금잔디·신유·송가인·박상철...', 'originallink': 'https://www.gukjenews.com/news/articleView.html?idxno=2572722', 'link': 'https://www.gukjenews.com/news/articleView.html?idxno=2572722', 'description': '보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자 한편, 가요무대는 매 방송마다 출연진 나이, 프로필, 근황 등이 관심을 모으고 있다.', 'pubDate': 'Mon, 17 Oct 2022 20:38:00 +0900'}, {'title': '[가요무대 출연진] 설운도·송가인·신유 출격...출연진과 선곡 목록은?', 'originallink': 'https://www.mhns.co.kr/news/articleView.html?idxno=536485', 'link': 'https://www.mhns.co.kr/news/articleView.html?idxno=536485', 'description': '짝사랑(고복수) / 양지원 16. 항구의 남자(1)+무조건(1절+후렴반복) / 박상철 17. 요즘여자 요즘남자(1)+오빠는 잘 있단다 / 현숙 18. 보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자', 'pubDate': 'Mon, 17 Oct 2022 20:02:00 +0900'}, {'title': '&apos;가요무대&apos; 출연진 설운도-김용임-진성-금잔디-송가인-박서진 등 20곡 소개', 'originallink': 'http://www.joygm.com/news/articleView.html?idxno=100180', 'link': 'http://www.joygm.com/news/articleView.html?idxno=100180', 'description': '<b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자 &apos;가요무대&apos; 오늘 &apos;출연진&apos;과 곡목 소개와 더불어 사회자인 김동건 아나운서에도 관심이 쏠리고 있는 바, 그는 1939년 출생으로 올해 나이 83세이다. 1964년... ', 'pubDate': 'Mon, 17 Oct 2022 15:08:00 +0900'}, {'title': '&apos;가요무대&apos; 오늘(17일) 설운도·진성·김용임·박상철·현숙·풍금·박서진', 'originallink': 'http://www.celuvmedia.com/article.php?aid=1665971909438721007', 'link': 'http://www.celuvmedia.com/article.php?aid=1665971909438721007', 'description': '잘 있단다&apos; 설운도 &apos;보랏빛 엽서&apos; 진성 &apos;<b>안동역</b>에서&apos; 여성 전 출연자 &apos;머나먼 고향&apos; 등 다채로운 무대가 공개된다. ‘가요무대’는 매주 월요일 오후 10시에 방송된다. [셀럽미디어 / 사진=&apos;가요무대&apos; 홈페이지]', 'pubDate': 'Mon, 17 Oct 2022 11:08:00 +0900'}, {'title': '김천시민의 날 기념, 시민화합대잔치 인산인해 대성황', 'originallink': 'http://www.kbsm.net/news/view.php?idx=367003', 'link': 'http://www.kbsm.net/news/view.php?idx=367003', 'description': '식후행사에는 인기가수 장윤정의 히트곡 사랑아와 진성의 <b>안동역</b> 열창에는 참석한 모든 시민이 환호하며 앵콜 제창과 인기가수 축하공연에서 국민 트로트 가수 김연자와 박 군이 출연해 십 분 내로, 한잔해 등 자신의... ', 'pubDate': 'Sun, 16 Oct 2022 16:24:00 +0900'}, {'title': '김천시민체육대회 개막 축하 공연하는 진성', 'originallink': 'http://www.newsis.com/view/?id=NISI20221014_0001107005', 'link': 'https://n.news.naver.com/mnews/article/003/0011476047?sid=102', 'description': '트로트 가수 진성이 14일 오후 경북 김천스포츠타운 보조경기장에서 열린 &apos;2022 김천시민체육대회&apos; 개막 축하 공연에서 자신의 히트곡 &apos;<b>안동역</b>&apos;을 부르고 있다. 2022.10.14 phs6431@newsis.com 공감언론 뉴시스가 독자... ', 'pubDate': 'Fri, 14 Oct 2022 20:56:00 +0900'}, {'title': '기자 출신 항일시인 이육사 기리는 기자상 제정된다', 'originallink': 'http://www.journalist.or.kr/news/article.html?no=52332', 'link': 'https://n.news.naver.com/mnews/article/127/0000033160?sid=102', 'description': '대구경북지역 출신 전직 언론인들이 주축이 된 이육사기자상 제정위원회(위원회)는 17일 오후 4시 경북 구 <b>안동역</b> 앞 경북문화콘텐츠진흥원 1층 창조아트홀에서 창립총회를 개최한다. 경북도립대학교와... ', 'pubDate': 'Fri, 14 Oct 2022 11:05:00 +0900'}, {'title': '&lt;특집&gt; 민선8기 권기창號 100일…‘안동시정 혁신’ 쾌조의 순항', 'originallink': 'http://www.ksmnews.co.kr/default/index_view_page.php?idx=397951&part_idx=243', 'link': 'http://www.ksmnews.co.kr/default/index_view_page.php?idx=397951&part_idx=243', 'description': '구)<b>안동역</b>은 야외워터파크와 키즈테마파크 등 문화관광타운으로 조성해 ‘첫눈이 오는 날’ 이벤트와 상설트롯 콘서트 등 차별화된 이벤트를 하며 버스터미널도 신설한다. 폐선부지는 마라톤코스, 자전거길, 트래킹 등... ', 'pubDate': 'Thu, 13 Oct 2022 22:04:00 +0900'}, {'title': '&quot;이육사 선생 투혼적 기자정신 이어가자&quot;', 'originallink': 'https://www.idaegu.co.kr/news/articleView.html?idxno=397598', 'link': 'https://www.idaegu.co.kr/news/articleView.html?idxno=397598', 'description': '이육사기자상 창립총회가 오는 17일 구 <b>안동역</b> 앞 경상북도콘텐츠진흥원에서 개최된다. 13일 김시묘 이육사기자상 제정위원장은 &quot;이육사 선생의 투혼적인 기자정신은 오늘날에도 바른 언론의 향도가 되기에 부족함이... ', 'pubDate': 'Thu, 13 Oct 2022 21:34:00 +0900'}, {'title': '진성, ‘미스터트롯2’ 마스터 합류 “전 시즌 출신 후배들이 활약하는 모습 ...', 'originallink': 'https://www.bntnews.co.kr/article/view/bnt202210130017', 'link': 'https://www.bntnews.co.kr/article/view/bnt202210130017', 'description': '특히 진성의 수많은 히트곡 ‘<b>안동역</b>에서’, ‘태클을 걸지마’, ‘동전인생’, ‘울엄마’, ‘가지마’, ‘님의 등불’, ‘보릿고개’ 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 진성은 “전 시즌 출신의... ', 'pubDate': 'Thu, 13 Oct 2022 11:12:00 +0900'}, {'title': '진성, &apos;미스터트롯2&apos; 마스터 합류…히트곡도 오디션곡으로 인기 폭발', 'originallink': 'http://www.psnews.co.kr/news/articleView.html?idxno=2015185', 'link': 'http://www.psnews.co.kr/news/articleView.html?idxno=2015185', 'description': '특히 진성의 수많은 히트곡 &apos;<b>안동역</b>에서&apos;, &apos;태클을 걸지마&apos;, &apos;동전인생&apos;, &apos;울엄마&apos;, &apos;가지마&apos;, &apos;님의 등불&apos;, &apos;보릿고개&apos; 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 진성 곡을 부른 참가자들의 결과가 좋아... ', 'pubDate': 'Thu, 13 Oct 2022 10:34:00 +0900'}, {'title': '진성, &apos;미스터트롯2&apos; 마스터 합류 &quot;전 시즌 출신 후배들 활약 보며 뿌듯함 느...', 'originallink': 'https://tenasia.hankyung.com/tv/article/2022101346244', 'link': 'https://n.news.naver.com/mnews/article/312/0000575037?sid=106', 'description': '특히 진성의 수많은 히트곡 ‘<b>안동역</b>에서’, ‘태클을 걸지마’, ‘동전인생’, ‘울엄마’, ‘가지마’, ‘님의 등불’, ‘보릿고개’ 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 이와 관련 진성은 “전... ', 'pubDate': 'Thu, 13 Oct 2022 09:59:00 +0900'}, {'title': '진성 ‘미스터트롯2’ 합류 “전 시즌 출신 후배들 활약 뿌듯함 느껴”', 'originallink': 'https://www.newsen.com/news_view.php?uid=202210130852211910', 'link': 'https://n.news.naver.com/mnews/article/609/0000640988?sid=106', 'description': '특히 진성의 히트곡 ‘<b>안동역</b>에서’, ‘태클을 걸지마’, ‘동전인생’, ‘울엄마’, ‘가지마’, ‘님의 등불’, ‘보릿고개’ 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 진성 곡을 부른 참가자들의... ', 'pubDate': 'Thu, 13 Oct 2022 08:55:00 +0900'}, {'title': '<b>안동역</b>에\xa0핀\xa0노래\xa0꽃…제1회\xa0김병걸\xa0가요제', 'originallink': 'http://www.smedaily.co.kr/news/articleView.html?idxno=240502', 'link': 'http://www.smedaily.co.kr/news/articleView.html?idxno=240502', 'description': '유차영 대중가요 평론가·한국콜마 연수원장 <b>안동역</b>\xa0광장에\xa0노래\xa0꽃이\xa0피었다.\xa0국민가요\xa0&lt;<b>안동역</b>에서&gt;\xa0노래\xa0탄생지,\xa0<b>안동역</b>\xa0광장에서\xa0펼쳐진,\xa0제1회\xa0김병걸가요제가\xa0그\xa0꽃떨기이다.\xa0한국예술인총연합회... ', 'pubDate': 'Wed, 12 Oct 2022 11:18:00 +0900'}, {'title': '[대구경북 초선 기초단체장] 권기창 안동시장 &quot;도청신도시 행정구역 통합, 인...', 'originallink': 'https://news.imaeil.com/page/view/2022100709462039321', 'link': 'https://n.news.naver.com/mnews/article/088/0000778892?sid=102', 'description': '권 시장은 &quot;옛 <b>안동역</b>은 야외워터파크와 키즈테마파크 등 문화관광타운으로 조성하고 버스터미널도 신설할 것&quot;이라며 &quot;폐선부지는 마라톤코스, 자전거길, 트래킹 등 복합레포츠단지로 개발하며, 간이역을 활용해 오감만족... ', 'pubDate': 'Wed, 12 Oct 2022 06:31:00 +0900'}, {'title': '&apos;이육사기자상&apos; 창립총회 개최…17일 오후 4시 경북콘텐츠진흥원', 'originallink': 'http://www.kyongbuk.co.kr/news/articleView.html?idxno=2114057', 'link': 'http://www.kyongbuk.co.kr/news/articleView.html?idxno=2114057', 'description': '이육사기자상 창립총회가 오는 17일 오후 4시부터 구 <b>안동역</b> 앞 경북콘텐츠진흥원 1층 창조아트홀에서 열린다. 이날 행사는 경북도청권 기자 및 전·현직 언론인 50여 명이 모여 결성한 이육사기자상 제정위원회가... ', 'pubDate': 'Tue, 11 Oct 2022 19:40:00 +0900'}, {'title': '‘미스터라디오’ 배기성, ‘<b>안동역</b>에서’ 라이브+깜짝 댄스 로 청취자 폭소 ...', 'originallink': 'http://sports.khan.co.kr/news/sk_index.html?art_id=202210111840003&sec_id=540201&pt=nv', 'link': 'https://n.news.naver.com/mnews/article/144/0000841706?sid=106', 'description': '배기성은 또 유리상자의 이세준과 팀 티키타카를 보인 가운데 쇼리와 음악 대결을 펼치기 위해 ‘<b>안동역</b>에서’를 열창, 명불허전 허스키 보이스와 깜짝 댄스 실력으로 청취자들에게 웃음을 선사했다. 이후 배기성은... ', 'pubDate': 'Tue, 11 Oct 2022 18:41:00 +0900'}]



```python
dic_res.keys()
```




    dict_keys(['lastBuildDate', 'total', 'start', 'display', 'items'])




```python
for item in dic_res['items']:
    print(item, end='\n\n')
```

    {'title': '&apos;가요무대&apos; 가수 설운도·박상철·진성, &quot;항구의 남자, 삼바의 여인&quot; 히트곡 ...', 'originallink': 'http://www.topstarnews.net/news/articleView.html?idxno=14759903', 'link': 'http://www.topstarnews.net/news/articleView.html?idxno=14759903', 'description': '설운도와 진성은 공연 막바지에 또 무대에 올라 각각 &apos;보랏빛 엽서&apos;와 &apos;<b>안동역</b>에서&apos;를 부르며 진한 감동을 전하기도 했다. KBS1 중장년층 대상 음악 프로그램 &apos;가요무대&apos;는 매주 월요일 오후 10시에 방송된다.... ', 'pubDate': 'Mon, 17 Oct 2022 22:06:00 +0900'}
    
    {'title': '&apos;가요무대&apos; 오늘(17일) 출연진 공개…송가인 박서진 현숙 박상철 설운도 등', 'originallink': 'http://sports.hankooki.com/news/articleView.html?idxno=6810002', 'link': 'http://sports.hankooki.com/news/articleView.html?idxno=6810002', 'description': '짝사랑(고복수) / 양지원 16. 항구의 남자(1)+무조건(1절+후렴반복) / 박상철 17. 요즘여자 요즘남자(1)+오빠는 잘 있단다 / 현숙 18. 보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자', 'pubDate': 'Mon, 17 Oct 2022 21:52:00 +0900'}
    
    {'title': '[가요무대 출연진] 가수 설운도·김용임·진성·금잔디·신유·송가인·박상철...', 'originallink': 'https://www.sisamagazine.co.kr/news/articleView.html?idxno=469943', 'link': 'https://www.sisamagazine.co.kr/news/articleView.html?idxno=469943', 'description': '보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자 한편, 가요무대는 매 방송마다 출연진 나이부터 프로필까지 시청자들의 관심을 끌고 있다.', 'pubDate': 'Mon, 17 Oct 2022 21:40:00 +0900'}
    
    {'title': '[가요무대 출연진] 가수 설운도·김용임·진성·금잔디·신유·송가인·박상철...', 'originallink': 'https://www.gukjenews.com/news/articleView.html?idxno=2572722', 'link': 'https://www.gukjenews.com/news/articleView.html?idxno=2572722', 'description': '보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자 한편, 가요무대는 매 방송마다 출연진 나이, 프로필, 근황 등이 관심을 모으고 있다.', 'pubDate': 'Mon, 17 Oct 2022 20:38:00 +0900'}
    
    {'title': '[가요무대 출연진] 설운도·송가인·신유 출격...출연진과 선곡 목록은?', 'originallink': 'https://www.mhns.co.kr/news/articleView.html?idxno=536485', 'link': 'https://www.mhns.co.kr/news/articleView.html?idxno=536485', 'description': '짝사랑(고복수) / 양지원 16. 항구의 남자(1)+무조건(1절+후렴반복) / 박상철 17. 요즘여자 요즘남자(1)+오빠는 잘 있단다 / 현숙 18. 보랏빛 엽서 / 설운도 19. <b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자', 'pubDate': 'Mon, 17 Oct 2022 20:02:00 +0900'}
    
    {'title': '&apos;가요무대&apos; 출연진 설운도-김용임-진성-금잔디-송가인-박서진 등 20곡 소개', 'originallink': 'http://www.joygm.com/news/articleView.html?idxno=100180', 'link': 'http://www.joygm.com/news/articleView.html?idxno=100180', 'description': '<b>안동역</b>에서 / 진성 20. 머나먼 고향(나훈아) / 여성 전 출연자 &apos;가요무대&apos; 오늘 &apos;출연진&apos;과 곡목 소개와 더불어 사회자인 김동건 아나운서에도 관심이 쏠리고 있는 바, 그는 1939년 출생으로 올해 나이 83세이다. 1964년... ', 'pubDate': 'Mon, 17 Oct 2022 15:08:00 +0900'}
    
    {'title': '&apos;가요무대&apos; 오늘(17일) 설운도·진성·김용임·박상철·현숙·풍금·박서진', 'originallink': 'http://www.celuvmedia.com/article.php?aid=1665971909438721007', 'link': 'http://www.celuvmedia.com/article.php?aid=1665971909438721007', 'description': '잘 있단다&apos; 설운도 &apos;보랏빛 엽서&apos; 진성 &apos;<b>안동역</b>에서&apos; 여성 전 출연자 &apos;머나먼 고향&apos; 등 다채로운 무대가 공개된다. ‘가요무대’는 매주 월요일 오후 10시에 방송된다. [셀럽미디어 / 사진=&apos;가요무대&apos; 홈페이지]', 'pubDate': 'Mon, 17 Oct 2022 11:08:00 +0900'}
    
    {'title': '김천시민의 날 기념, 시민화합대잔치 인산인해 대성황', 'originallink': 'http://www.kbsm.net/news/view.php?idx=367003', 'link': 'http://www.kbsm.net/news/view.php?idx=367003', 'description': '식후행사에는 인기가수 장윤정의 히트곡 사랑아와 진성의 <b>안동역</b> 열창에는 참석한 모든 시민이 환호하며 앵콜 제창과 인기가수 축하공연에서 국민 트로트 가수 김연자와 박 군이 출연해 십 분 내로, 한잔해 등 자신의... ', 'pubDate': 'Sun, 16 Oct 2022 16:24:00 +0900'}
    
    {'title': '김천시민체육대회 개막 축하 공연하는 진성', 'originallink': 'http://www.newsis.com/view/?id=NISI20221014_0001107005', 'link': 'https://n.news.naver.com/mnews/article/003/0011476047?sid=102', 'description': '트로트 가수 진성이 14일 오후 경북 김천스포츠타운 보조경기장에서 열린 &apos;2022 김천시민체육대회&apos; 개막 축하 공연에서 자신의 히트곡 &apos;<b>안동역</b>&apos;을 부르고 있다. 2022.10.14 phs6431@newsis.com 공감언론 뉴시스가 독자... ', 'pubDate': 'Fri, 14 Oct 2022 20:56:00 +0900'}
    
    {'title': '기자 출신 항일시인 이육사 기리는 기자상 제정된다', 'originallink': 'http://www.journalist.or.kr/news/article.html?no=52332', 'link': 'https://n.news.naver.com/mnews/article/127/0000033160?sid=102', 'description': '대구경북지역 출신 전직 언론인들이 주축이 된 이육사기자상 제정위원회(위원회)는 17일 오후 4시 경북 구 <b>안동역</b> 앞 경북문화콘텐츠진흥원 1층 창조아트홀에서 창립총회를 개최한다. 경북도립대학교와... ', 'pubDate': 'Fri, 14 Oct 2022 11:05:00 +0900'}
    
    {'title': '&lt;특집&gt; 민선8기 권기창號 100일…‘안동시정 혁신’ 쾌조의 순항', 'originallink': 'http://www.ksmnews.co.kr/default/index_view_page.php?idx=397951&part_idx=243', 'link': 'http://www.ksmnews.co.kr/default/index_view_page.php?idx=397951&part_idx=243', 'description': '구)<b>안동역</b>은 야외워터파크와 키즈테마파크 등 문화관광타운으로 조성해 ‘첫눈이 오는 날’ 이벤트와 상설트롯 콘서트 등 차별화된 이벤트를 하며 버스터미널도 신설한다. 폐선부지는 마라톤코스, 자전거길, 트래킹 등... ', 'pubDate': 'Thu, 13 Oct 2022 22:04:00 +0900'}
    
    {'title': '&quot;이육사 선생 투혼적 기자정신 이어가자&quot;', 'originallink': 'https://www.idaegu.co.kr/news/articleView.html?idxno=397598', 'link': 'https://www.idaegu.co.kr/news/articleView.html?idxno=397598', 'description': '이육사기자상 창립총회가 오는 17일 구 <b>안동역</b> 앞 경상북도콘텐츠진흥원에서 개최된다. 13일 김시묘 이육사기자상 제정위원장은 &quot;이육사 선생의 투혼적인 기자정신은 오늘날에도 바른 언론의 향도가 되기에 부족함이... ', 'pubDate': 'Thu, 13 Oct 2022 21:34:00 +0900'}
    
    {'title': '진성, ‘미스터트롯2’ 마스터 합류 “전 시즌 출신 후배들이 활약하는 모습 ...', 'originallink': 'https://www.bntnews.co.kr/article/view/bnt202210130017', 'link': 'https://www.bntnews.co.kr/article/view/bnt202210130017', 'description': '특히 진성의 수많은 히트곡 ‘<b>안동역</b>에서’, ‘태클을 걸지마’, ‘동전인생’, ‘울엄마’, ‘가지마’, ‘님의 등불’, ‘보릿고개’ 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 진성은 “전 시즌 출신의... ', 'pubDate': 'Thu, 13 Oct 2022 11:12:00 +0900'}
    
    {'title': '진성, &apos;미스터트롯2&apos; 마스터 합류…히트곡도 오디션곡으로 인기 폭발', 'originallink': 'http://www.psnews.co.kr/news/articleView.html?idxno=2015185', 'link': 'http://www.psnews.co.kr/news/articleView.html?idxno=2015185', 'description': '특히 진성의 수많은 히트곡 &apos;<b>안동역</b>에서&apos;, &apos;태클을 걸지마&apos;, &apos;동전인생&apos;, &apos;울엄마&apos;, &apos;가지마&apos;, &apos;님의 등불&apos;, &apos;보릿고개&apos; 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 진성 곡을 부른 참가자들의 결과가 좋아... ', 'pubDate': 'Thu, 13 Oct 2022 10:34:00 +0900'}
    
    {'title': '진성, &apos;미스터트롯2&apos; 마스터 합류 &quot;전 시즌 출신 후배들 활약 보며 뿌듯함 느...', 'originallink': 'https://tenasia.hankyung.com/tv/article/2022101346244', 'link': 'https://n.news.naver.com/mnews/article/312/0000575037?sid=106', 'description': '특히 진성의 수많은 히트곡 ‘<b>안동역</b>에서’, ‘태클을 걸지마’, ‘동전인생’, ‘울엄마’, ‘가지마’, ‘님의 등불’, ‘보릿고개’ 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 이와 관련 진성은 “전... ', 'pubDate': 'Thu, 13 Oct 2022 09:59:00 +0900'}
    
    {'title': '진성 ‘미스터트롯2’ 합류 “전 시즌 출신 후배들 활약 뿌듯함 느껴”', 'originallink': 'https://www.newsen.com/news_view.php?uid=202210130852211910', 'link': 'https://n.news.naver.com/mnews/article/609/0000640988?sid=106', 'description': '특히 진성의 히트곡 ‘<b>안동역</b>에서’, ‘태클을 걸지마’, ‘동전인생’, ‘울엄마’, ‘가지마’, ‘님의 등불’, ‘보릿고개’ 등은 오디션 프로그램 필수 선곡 리스트로 사랑받고 있다. 진성 곡을 부른 참가자들의... ', 'pubDate': 'Thu, 13 Oct 2022 08:55:00 +0900'}
    
    {'title': '<b>안동역</b>에\xa0핀\xa0노래\xa0꽃…제1회\xa0김병걸\xa0가요제', 'originallink': 'http://www.smedaily.co.kr/news/articleView.html?idxno=240502', 'link': 'http://www.smedaily.co.kr/news/articleView.html?idxno=240502', 'description': '유차영 대중가요 평론가·한국콜마 연수원장 <b>안동역</b>\xa0광장에\xa0노래\xa0꽃이\xa0피었다.\xa0국민가요\xa0&lt;<b>안동역</b>에서&gt;\xa0노래\xa0탄생지,\xa0<b>안동역</b>\xa0광장에서\xa0펼쳐진,\xa0제1회\xa0김병걸가요제가\xa0그\xa0꽃떨기이다.\xa0한국예술인총연합회... ', 'pubDate': 'Wed, 12 Oct 2022 11:18:00 +0900'}
    
    {'title': '[대구경북 초선 기초단체장] 권기창 안동시장 &quot;도청신도시 행정구역 통합, 인...', 'originallink': 'https://news.imaeil.com/page/view/2022100709462039321', 'link': 'https://n.news.naver.com/mnews/article/088/0000778892?sid=102', 'description': '권 시장은 &quot;옛 <b>안동역</b>은 야외워터파크와 키즈테마파크 등 문화관광타운으로 조성하고 버스터미널도 신설할 것&quot;이라며 &quot;폐선부지는 마라톤코스, 자전거길, 트래킹 등 복합레포츠단지로 개발하며, 간이역을 활용해 오감만족... ', 'pubDate': 'Wed, 12 Oct 2022 06:31:00 +0900'}
    
    {'title': '&apos;이육사기자상&apos; 창립총회 개최…17일 오후 4시 경북콘텐츠진흥원', 'originallink': 'http://www.kyongbuk.co.kr/news/articleView.html?idxno=2114057', 'link': 'http://www.kyongbuk.co.kr/news/articleView.html?idxno=2114057', 'description': '이육사기자상 창립총회가 오는 17일 오후 4시부터 구 <b>안동역</b> 앞 경북콘텐츠진흥원 1층 창조아트홀에서 열린다. 이날 행사는 경북도청권 기자 및 전·현직 언론인 50여 명이 모여 결성한 이육사기자상 제정위원회가... ', 'pubDate': 'Tue, 11 Oct 2022 19:40:00 +0900'}
    
    {'title': '‘미스터라디오’ 배기성, ‘<b>안동역</b>에서’ 라이브+깜짝 댄스 로 청취자 폭소 ...', 'originallink': 'http://sports.khan.co.kr/news/sk_index.html?art_id=202210111840003&sec_id=540201&pt=nv', 'link': 'https://n.news.naver.com/mnews/article/144/0000841706?sid=106', 'description': '배기성은 또 유리상자의 이세준과 팀 티키타카를 보인 가운데 쇼리와 음악 대결을 펼치기 위해 ‘<b>안동역</b>에서’를 열창, 명불허전 허스키 보이스와 깜짝 댄스 실력으로 청취자들에게 웃음을 선사했다. 이후 배기성은... ', 'pubDate': 'Tue, 11 Oct 2022 18:41:00 +0900'}
    



```python
print( type( dic_res['items'] ) , end="\n\n")
print( dic_res['items'][0], end="\n\n" )
print( dic_res['items'][0].keys())
```

    <class 'list'>
    
    {'title': '&apos;가요무대&apos; 가수 설운도·박상철·진성, &quot;항구의 남자, 삼바의 여인&quot; 히트곡 ...', 'originallink': 'http://www.topstarnews.net/news/articleView.html?idxno=14759903', 'link': 'http://www.topstarnews.net/news/articleView.html?idxno=14759903', 'description': '설운도와 진성은 공연 막바지에 또 무대에 올라 각각 &apos;보랏빛 엽서&apos;와 &apos;<b>안동역</b>에서&apos;를 부르며 진한 감동을 전하기도 했다. KBS1 중장년층 대상 음악 프로그램 &apos;가요무대&apos;는 매주 월요일 오후 10시에 방송된다.... ', 'pubDate': 'Mon, 17 Oct 2022 22:06:00 +0900'}
    
    dict_keys(['title', 'originallink', 'link', 'description', 'pubDate'])


## 카페 글 검색
[카페 검색에 대한 설명](https://developers.naver.com/docs/serviceapi/search/cafearticle/cafearticle.md#%EC%B9%B4%ED%8E%98%EA%B8%80)

과제 - 카페글을 검색하는데 검색 키워드는 임의로 정해서 검색어와 유사성이 높은 글 50개를 뽑으시오.

## 웹 문서 검색
[웹 문서 검색에 대한 설명](https://developers.naver.com/docs/serviceapi/search/web/web.md#%EC%9B%B9%EB%AC%B8%EC%84%9C)

## 지역 검색
네이버(지역 서비스)에 등록된 각 지역별 업체 및 상호 검색

# 검색어 트랜드
[관련 설명](https://developers.naver.com/docs/serviceapi/datalab/search/search.md#%ED%86%B5%ED%95%A9-%EA%B2%80%EC%83%89%EC%96%B4-%ED%8A%B8%EB%A0%8C%EB%93%9C-%EA%B0%9C%EC%9A%94)

검색 추이 데이터를 json 형태로 반환.  
**통합 검색어 트렌드 API의 하루 호출 한도는 1,000회입니다.**

[여기](https://developers.naver.com/apps/#/list) 접속


```python
import os
import sys
import urllib.request
client_id = "eGaTUY3HVt4LzOMY6fWT"
client_secret = "klCa7Y07ml"
url = "https://openapi.naver.com/v1/datalab/search";
body = "{\"startDate\":\"2017-01-01\",\"endDate\":\"2017-04-30\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"한글\",\"keywords\":[\"한글\",\"korean\"]},{\"groupName\":\"영어\",\"keywords\":[\"영어\",\"english\"]}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}";

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
```

    {"startDate":"2017-01-01","endDate":"2017-04-30","timeUnit":"month","results":[{"title":"한글","keywords":["한글","korean"],"data":[{"period":"2017-01-01","ratio":47.00101},{"period":"2017-02-01","ratio":53.23619},{"period":"2017-03-01","ratio":100},{"period":"2017-04-01","ratio":85.327}]},{"title":"영어","keywords":["영어","english"],"data":[{"period":"2017-01-01","ratio":40.0881},{"period":"2017-02-01","ratio":36.69942},{"period":"2017-03-01","ratio":52.11792},{"period":"2017-04-01","ratio":44.4595}]}]}



```python
type(eval(response_body.decode('utf-8')))
rData = eval(response_body.decode('utf-8'))
rData
```




    {'startDate': '2017-01-01',
     'endDate': '2017-04-30',
     'timeUnit': 'month',
     'results': [{'title': '한글',
       'keywords': ['한글', 'korean'],
       'data': [{'period': '2017-01-01', 'ratio': 47.00101},
        {'period': '2017-02-01', 'ratio': 53.23619},
        {'period': '2017-03-01', 'ratio': 100},
        {'period': '2017-04-01', 'ratio': 85.327}]},
      {'title': '영어',
       'keywords': ['영어', 'english'],
       'data': [{'period': '2017-01-01', 'ratio': 40.0881},
        {'period': '2017-02-01', 'ratio': 36.69942},
        {'period': '2017-03-01', 'ratio': 52.11792},
        {'period': '2017-04-01', 'ratio': 44.4595}]}]}




```python
import os
import sys
import urllib.request
client_id = "_70FX9MPd8HWnJBxd2_S"
client_secret = "mKjz06XPAy"
url = "https://openapi.naver.com/v1/datalab/search";
body = "{\"startDate\":\"2021-01-01\",\"endDate\":\"2021-12-31\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"한글\",\"keywords\":[\"한글\",\"korean\"]}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}";

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
    
    type(eval(response_body.decode('utf-8')))
rData = eval(response_body.decode('utf-8'))
rData
```

    {"startDate":"2021-01-01","endDate":"2021-12-31","timeUnit":"month","results":[{"title":"한글","keywords":["한글","korean"],"data":[{"period":"2021-01-01","ratio":47.5716},{"period":"2021-02-01","ratio":35.94852},{"period":"2021-03-01","ratio":100},{"period":"2021-04-01","ratio":76.13117},{"period":"2021-05-01","ratio":96.6376},{"period":"2021-06-01","ratio":77.21046},{"period":"2021-07-01","ratio":64.84018},{"period":"2021-08-01","ratio":59.94188},{"period":"2021-09-01","ratio":78.8709},{"period":"2021-10-01","ratio":89.74678},{"period":"2021-11-01","ratio":79.36903},{"period":"2021-12-01","ratio":54.21336}]}]}





    {'startDate': '2021-01-01',
     'endDate': '2021-12-31',
     'timeUnit': 'month',
     'results': [{'title': '한글',
       'keywords': ['한글', 'korean'],
       'data': [{'period': '2021-01-01', 'ratio': 47.5716},
        {'period': '2021-02-01', 'ratio': 35.94852},
        {'period': '2021-03-01', 'ratio': 100},
        {'period': '2021-04-01', 'ratio': 76.13117},
        {'period': '2021-05-01', 'ratio': 96.6376},
        {'period': '2021-06-01', 'ratio': 77.21046},
        {'period': '2021-07-01', 'ratio': 64.84018},
        {'period': '2021-08-01', 'ratio': 59.94188},
        {'period': '2021-09-01', 'ratio': 78.8709},
        {'period': '2021-10-01', 'ratio': 89.74678},
        {'period': '2021-11-01', 'ratio': 79.36903},
        {'period': '2021-12-01', 'ratio': 54.21336}]}]}




```python
client_id = "_70FX9MPd8HWnJBxd2_S"
client_secret = "mKjz06XPAy"
url = "https://openapi.naver.com/v1/datalab/search";
body = "{\"startDate\":\"2021-01-01\",\"endDate\":\"2021-12-31\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"크리스마스\",\"keywords\":[\"크리스마스\",\"성탄절\"]}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}";

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
    
    type(eval(response_body.decode('utf-8')))
rData = eval(response_body.decode('utf-8'))
rData
```

    {"startDate":"2021-01-01","endDate":"2021-12-31","timeUnit":"month","results":[{"title":"크리스마스","keywords":["크리스마스","성탄절"],"data":[{"period":"2021-01-01","ratio":3.47593},{"period":"2021-02-01","ratio":1.66598},{"period":"2021-03-01","ratio":2.13903},{"period":"2021-04-01","ratio":2.28301},{"period":"2021-05-01","ratio":4.29864},{"period":"2021-06-01","ratio":3.20855},{"period":"2021-07-01","ratio":3.04401},{"period":"2021-08-01","ratio":4.15466},{"period":"2021-09-01","ratio":6.89016},{"period":"2021-10-01","ratio":14.37679},{"period":"2021-11-01","ratio":40.5183},{"period":"2021-12-01","ratio":100}]}]}





    {'startDate': '2021-01-01',
     'endDate': '2021-12-31',
     'timeUnit': 'month',
     'results': [{'title': '크리스마스',
       'keywords': ['크리스마스', '성탄절'],
       'data': [{'period': '2021-01-01', 'ratio': 3.47593},
        {'period': '2021-02-01', 'ratio': 1.66598},
        {'period': '2021-03-01', 'ratio': 2.13903},
        {'period': '2021-04-01', 'ratio': 2.28301},
        {'period': '2021-05-01', 'ratio': 4.29864},
        {'period': '2021-06-01', 'ratio': 3.20855},
        {'period': '2021-07-01', 'ratio': 3.04401},
        {'period': '2021-08-01', 'ratio': 4.15466},
        {'period': '2021-09-01', 'ratio': 6.89016},
        {'period': '2021-10-01', 'ratio': 14.37679},
        {'period': '2021-11-01', 'ratio': 40.5183},
        {'period': '2021-12-01', 'ratio': 100}]}]}



## 과제 - 검색율을 월별로 그래프로 그리시오.


```python
client_id = "_70FX9MPd8HWnJBxd2_S"
client_secret = "mKjz06XPAy"
url = "https://openapi.naver.com/v1/datalab/search";
body = "{\"startDate\":\"2021-01-01\",\"endDate\":\"2021-12-31\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"할로윈\",\"keywords\":[\"할로윈\",\"Halloween\"]}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}";

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
request.add_header("Content-Type","application/json")
response = urllib.request.urlopen(request, data=body.encode("utf-8"))
rescode = response.getcode()
```


```python
rData = eval(response_body.decode('utf-8'))
```


```python
rData
```




    {'startDate': '2021-01-01',
     'endDate': '2021-12-31',
     'timeUnit': 'month',
     'results': [{'title': '크리스마스',
       'keywords': ['크리스마스', '성탄절'],
       'data': [{'period': '2021-01-01', 'ratio': 3.47593},
        {'period': '2021-02-01', 'ratio': 1.66598},
        {'period': '2021-03-01', 'ratio': 2.13903},
        {'period': '2021-04-01', 'ratio': 2.28301},
        {'period': '2021-05-01', 'ratio': 4.29864},
        {'period': '2021-06-01', 'ratio': 3.20855},
        {'period': '2021-07-01', 'ratio': 3.04401},
        {'period': '2021-08-01', 'ratio': 4.15466},
        {'period': '2021-09-01', 'ratio': 6.89016},
        {'period': '2021-10-01', 'ratio': 14.37679},
        {'period': '2021-11-01', 'ratio': 40.5183},
        {'period': '2021-12-01', 'ratio': 100}]}]}




```python
rData.keys()
```




    dict_keys(['startDate', 'endDate', 'timeUnit', 'results'])




```python
ratio_len = len(rData['results'][0]['data'])
```


```python
rData['results'][0]['data']
```




    [{'period': '2021-01-01', 'ratio': 3.47593},
     {'period': '2021-02-01', 'ratio': 1.66598},
     {'period': '2021-03-01', 'ratio': 2.13903},
     {'period': '2021-04-01', 'ratio': 2.28301},
     {'period': '2021-05-01', 'ratio': 4.29864},
     {'period': '2021-06-01', 'ratio': 3.20855},
     {'period': '2021-07-01', 'ratio': 3.04401},
     {'period': '2021-08-01', 'ratio': 4.15466},
     {'period': '2021-09-01', 'ratio': 6.89016},
     {'period': '2021-10-01', 'ratio': 14.37679},
     {'period': '2021-11-01', 'ratio': 40.5183},
     {'period': '2021-12-01', 'ratio': 100}]




```python
data = rData['results'][0]['data']
```


```python
import matplotlib.pyplot as plt

x=[]
y=[]

plt.figure(figsize=(12,4))
for i in range(0, len(data)):
    x.append(data[i]['period'].split('-')[1])
    y.append(data[i]['ratio'])
    plt.plot(x,y)
plt.show()
```

<img width="713" alt="스크린샷 2022-10-18 오후 5 41 14" src="https://user-images.githubusercontent.com/87309905/196381352-61a60f67-f8c3-40bb-a216-4608f118a73a.png">

    
    


---
오늘 마지막 (도전) 과제 - 쿠팡에서 특정 상품을 검색하고 해당 상품에 대한 고객 평을 수집하여 파일로 저장


```python
print(os.getcwd())
```

    /Users/ryu



```python
/Applications
```


```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import subprocess
import shutil

subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chrometemp"')

options = webdriver.ChromeOptions()
options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

driver = webdriver.Chrome('chromedriver.exe', chrome_options=options)

driver.implicitly_wait(6)

url = 'https://www.coupang.com'
driver.get(url)
```
