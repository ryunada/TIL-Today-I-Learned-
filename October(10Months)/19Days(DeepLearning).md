```python
import os
print(os.getcwd())
```

    /Users


---

# NLP(Natural Language Processing, 자연어 처리)
    - 텍스트에서 의미 있는 정보를 분석, 추출하고 이해하는 일련의 기술 집합

## KoNLPy : 파이썬 한국어 NLP 패키지
KoNLPy : "코엔엘파이"라고 읽음.
[참고](https://konlpy.org/ko/latest/#start)

### KoNLPY 설치 전 준비 사항
- 운영체제 확인
    * 시작 버튼 우클릭 --> 시스템 선택
        - **시스템 종류 : 64비트 운영 체제, x64 기반 프로세서**
- 파이썬 버전 확인
    * cmd 창에서 ```python --version```
        - **Python 3.10.8**
        
운영체제 비트 수와 파이썬 비트 수가 일치해야함.


```python
# 설치된 python이 몇 bit 버전인지 확인하는 코드
import platform
print( platform.architecture() )
```

    ('64bit', '')


- Java 버전 확인(1.7 이상이어야 함.)
    * cmd 창에서 ``javac -version```
        - **javac 11.0.11** 
        - 설치 필요할 때 [여기 클릭](https://www.oracle.com/java/technologies/downloads/#jdk19-windows)  
          이참에 최신 버전으로 하나 설치하죠.
- 환경 변수 등록(ppt 참고)
- JPype 설치(ppt 참고)


```python
conda install -c conda-forge jpype1

```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/anaconda3/envs/tensorflow2
    
      added / updated specs:
        - jpype1
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        ca-certificates-2022.9.24  |       h033912b_0         150 KB  conda-forge
        certifi-2022.9.24          |     pyhd8ed1ab_0         155 KB  conda-forge
        jpype1-1.4.0               |  py310habb735a_0         412 KB  conda-forge
        openssl-1.1.1q             |       hfe4f2af_0         1.9 MB  conda-forge
        python_abi-3.10            |          2_cp310           4 KB  conda-forge
        ------------------------------------------------------------
                                               Total:         2.6 MB
    
    The following NEW packages will be INSTALLED:
    
      jpype1             conda-forge/osx-64::jpype1-1.4.0-py310habb735a_0 None
      python_abi         conda-forge/osx-64::python_abi-3.10-2_cp310 None
    
    The following packages will be UPDATED:
    
      ca-certificates    pkgs/main::ca-certificates-2022.07.19~ --> conda-forge::ca-certificates-2022.9.24-h033912b_0 None
    
    The following packages will be SUPERSEDED by a higher-priority channel:
    
      certifi            pkgs/main/osx-64::certifi-2022.9.24-p~ --> conda-forge/noarch::certifi-2022.9.24-pyhd8ed1ab_0 None
      openssl              pkgs/main::openssl-1.1.1q-hca72f7f_0 --> conda-forge::openssl-1.1.1q-hfe4f2af_0 None
    
    
    
    Downloading and Extracting Packages
    jpype1-1.4.0         | 412 KB    | ##################################### | 100% 
    python_abi-3.10      | 4 KB      | ##################################### | 100% 
    ca-certificates-2022 | 150 KB    | ##################################### | 100% 
    certifi-2022.9.24    | 155 KB    | ##################################### | 100% 
    openssl-1.1.1q       | 1.9 MB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Retrieving notices: ...working... done
    
    Note: you may need to restart the kernel to use updated packages.



```python
pip install konlpy
```

    Collecting konlpy
      Using cached konlpy-0.6.0-py2.py3-none-any.whl (19.4 MB)
    Requirement already satisfied: numpy>=1.6 in /opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages (from konlpy) (1.22.3)
    Requirement already satisfied: JPype1>=0.7.0 in /opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages (from konlpy) (1.4.0)
    Collecting lxml>=4.1.0
      Using cached lxml-4.9.1.tar.gz (3.4 MB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hBuilding wheels for collected packages: lxml
      Building wheel for lxml (setup.py) ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mpython setup.py bdist_wheel[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[209 lines of output][0m
      [31m   [0m Building lxml version 4.9.1.
      [31m   [0m Building without Cython.
      [31m   [0m Building against libxml2 2.9.4 and libxslt 1.1.29
      [31m   [0m running bdist_wheel
      [31m   [0m running build
      [31m   [0m running build_py
      [31m   [0m creating build
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/_elementpath.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/sax.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/pyclasslookup.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/__init__.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/builder.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/doctestcompare.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/usedoctest.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/cssselect.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/ElementInclude.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/__init__.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/soupparser.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/defs.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/_setmixin.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/clean.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/_diffcommand.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/html5parser.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/__init__.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/formfill.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/builder.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/ElementSoup.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/_html5builder.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/usedoctest.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m copying src/lxml/html/diff.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/html
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron
      [31m   [0m copying src/lxml/isoschematron/__init__.py -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron
      [31m   [0m copying src/lxml/etree.h -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/etree_api.h -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/lxml.etree.h -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/lxml.etree_api.h -> build/lib.macosx-10.9-x86_64-cpython-310/lxml
      [31m   [0m copying src/lxml/includes/xmlerror.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/c14n.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/xmlschema.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/__init__.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/schematron.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/tree.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/uri.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/etreepublic.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/xpath.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/htmlparser.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/xslt.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/config.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/xmlparser.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/xinclude.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/dtdvalid.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/relaxng.pxd -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/lxml-version.h -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m copying src/lxml/includes/etree_defs.h -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/rng
      [31m   [0m copying src/lxml/isoschematron/resources/rng/iso-schematron.rng -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/rng
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/XSD2Schtrn.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/RNG2Schtrn.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl
      [31m   [0m creating build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_abstract_expand.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_dsdl_include.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_schematron_skeleton_for_xslt1.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_svrl_for_xslt1.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_schematron_message.xsl -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying src/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/readme.txt -> build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m running build_ext
      [31m   [0m building 'lxml.etree' extension
      [31m   [0m creating build/temp.macosx-10.9-x86_64-cpython-310
      [31m   [0m creating build/temp.macosx-10.9-x86_64-cpython-310/src
      [31m   [0m creating build/temp.macosx-10.9-x86_64-cpython-310/src/lxml
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/etree.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/etree.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/etree.o -lxslt -lexslt -lxml2 -lz -lm -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/etree.cpython-310-darwin.so -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
      [31m   [0m building 'lxml.objectify' extension
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/objectify.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/objectify.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/objectify.o -lxslt -lexslt -lxml2 -lz -lm -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/objectify.cpython-310-darwin.so -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
      [31m   [0m building 'lxml.builder' extension
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/builder.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/builder.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/builder.o -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/builder.cpython-310-darwin.so
      [31m   [0m building 'lxml._elementpath' extension
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/_elementpath.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/_elementpath.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/_elementpath.o -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/_elementpath.cpython-310-darwin.so
      [31m   [0m building 'lxml.html.diff' extension
      [31m   [0m creating build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/html
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/html/diff.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/html/diff.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/html/diff.o -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/diff.cpython-310-darwin.so
      [31m   [0m building 'lxml.html.clean' extension
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/html/clean.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/html/clean.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/html/clean.o -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/clean.cpython-310-darwin.so
      [31m   [0m building 'lxml.sax' extension
      [31m   [0m clang -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -fPIC -O2 -isystem /opt/anaconda3/envs/tensorflow2/include -DCYTHON_CLINE_IN_TRACEBACK=0 -Isrc -Isrc/lxml/includes -I/opt/anaconda3/envs/tensorflow2/include/python3.10 -c src/lxml/sax.c -o build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/sax.o -w -flat_namespace
      [31m   [0m clang -bundle -undefined dynamic_lookup -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib -Wl,-rpath,/opt/anaconda3/envs/tensorflow2/lib -L/opt/anaconda3/envs/tensorflow2/lib build/temp.macosx-10.9-x86_64-cpython-310/src/lxml/sax.o -o build/lib.macosx-10.9-x86_64-cpython-310/lxml/sax.cpython-310-darwin.so
      [31m   [0m /opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/command/install.py:34: SetuptoolsDeprecationWarning: setup.py install is deprecated. Use build and pip and other standards-based tools.
      [31m   [0m   warnings.warn(
      [31m   [0m installing to build/bdist.macosx-10.9-x86_64/wheel
      [31m   [0m running install
      [31m   [0m running install_lib
      [31m   [0m creating build/bdist.macosx-10.9-x86_64
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/_elementpath.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/XSD2Schtrn.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_abstract_expand.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_dsdl_include.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_schematron_skeleton_for_xslt1.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_svrl_for_xslt1.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/iso_schematron_message.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/iso-schematron-xslt1/readme.txt -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl/iso-schematron-xslt1
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/xsl/RNG2Schtrn.xsl -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/xsl
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/rng
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/resources/rng/iso-schematron.rng -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron/resources/rng
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/isoschematron/__init__.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/isoschematron
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/etree.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/objectify.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/xmlerror.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/c14n.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/xmlschema.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/__init__.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/lxml-version.h -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/schematron.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/__init__.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/tree.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/uri.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/etree_defs.h -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/etreepublic.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/xpath.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/htmlparser.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/xslt.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/config.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/xmlparser.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/xinclude.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/dtdvalid.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/includes/relaxng.pxd -> build/bdist.macosx-10.9-x86_64/wheel/lxml/includes
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/sax.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/pyclasslookup.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/__init__.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m creating build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/soupparser.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/defs.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/_setmixin.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/clean.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/_diffcommand.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/html5parser.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/__init__.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/formfill.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/builder.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/ElementSoup.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/_html5builder.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/clean.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/usedoctest.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/diff.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/html/diff.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml/html
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/etree_api.h -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/builder.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/builder.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/sax.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/doctestcompare.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/usedoctest.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/cssselect.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/_elementpath.cpython-310-darwin.so -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/lxml.etree.h -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/etree.h -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/ElementInclude.py -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m copying build/lib.macosx-10.9-x86_64-cpython-310/lxml/lxml.etree_api.h -> build/bdist.macosx-10.9-x86_64/wheel/lxml
      [31m   [0m running install_egg_info
      [31m   [0m running egg_info
      [31m   [0m writing src/lxml.egg-info/PKG-INFO
      [31m   [0m writing dependency_links to src/lxml.egg-info/dependency_links.txt
      [31m   [0m writing requirements to src/lxml.egg-info/requires.txt
      [31m   [0m writing top-level names to src/lxml.egg-info/top_level.txt
      [31m   [0m reading manifest file 'src/lxml.egg-info/SOURCES.txt'
      [31m   [0m reading manifest template 'MANIFEST.in'
      [31m   [0m adding license file 'LICENSE.txt'
      [31m   [0m adding license file 'LICENSES.txt'
      [31m   [0m writing manifest file 'src/lxml.egg-info/SOURCES.txt'
      [31m   [0m Copying src/lxml.egg-info to build/bdist.macosx-10.9-x86_64/wheel/lxml-4.9.1-py3.10.egg-info
      [31m   [0m running install_scripts
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 34, in <module>
      [31m   [0m   File "/private/var/folders/hf/9cldw65x7j71qr4hjbw44yjc0000gn/T/pip-install-t0vlnt3c/lxml_2f31bd06bd6e442b81d815e54ccb7cc1/setup.py", line 204, in <module>
      [31m   [0m     setup(
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/__init__.py", line 87, in setup
      [31m   [0m     return distutils.core.setup(**attrs)
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 185, in setup
      [31m   [0m     return run_commands(dist)
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 201, in run_commands
      [31m   [0m     dist.run_commands()
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 973, in run_commands
      [31m   [0m     self.run_command(cmd)
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/dist.py", line 1217, in run_command
      [31m   [0m     super().run_command(command)
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 992, in run_command
      [31m   [0m     cmd_obj.run()
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/wheel/bdist_wheel.py", line 328, in run
      [31m   [0m     impl_tag, abi_tag, plat_tag = self.get_tag()
      [31m   [0m   File "/opt/anaconda3/envs/tensorflow2/lib/python3.10/site-packages/wheel/bdist_wheel.py", line 278, in get_tag
      [31m   [0m     assert tag in supported_tags, "would build wheel with unsupported tag {}".format(tag)
      [31m   [0m AssertionError: would build wheel with unsupported tag ('cp310', 'cp310', 'macosx_10_9_x86_64')
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [31m  ERROR: Failed building wheel for lxml[0m[31m
    [0m[?25h  Running setup.py clean for lxml
    Failed to build lxml
    Installing collected packages: lxml, konlpy
      Running setup.py install for lxml ... [?25ldone
    [33m  DEPRECATION: lxml was installed using the legacy 'setup.py install' method, because a wheel could not be built for it. A possible replacement is to fix the wheel build issue reported above. Discussion can be found at https://github.com/pypa/pip/issues/8368[0m[33m
    [0m[?25hSuccessfully installed konlpy-0.6.0 lxml-4.9.1
    Note: you may need to restart the kernel to use updated packages.



```python
from konlpy.tag import Kkma
```


```python
kkma = Kkma()
```


```python
# 문장 구분, 분리
print(kkma.sentences('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    ['한글 분석을 시작합니다.', '잘 되겠죠?']



```python
# 명사 구분, 분리
print(kkma.nouns('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    ['한글', '분석']



```python
print(kkma.pos('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    [('한글', 'NNG'), ('분석', 'NNG'), ('을', 'JKO'), ('시작하', 'VV'), ('ㅂ니다', 'EFN'), ('.', 'SF'), ('잘', 'MAG'), ('되', 'VV'), ('겠', 'EPT'), ('죠', 'EFN'), ('?', 'SF')]



```python
from konlpy.tag import Hannanum
hunnanum = Hannanum()
```


```python
# 문장 구분, 분리
print(hunnanum.nouns('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    ['한글', '분석', '시작']



```python
print(hunnanum.morphs('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    ['한글', '분석', '을', '시작', '하', 'ㅂ니다', '.', '잘', '되', '겠죠', '?']



```python
print(hunnanum.pos('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    [('한글', 'N'), ('분석', 'N'), ('을', 'J'), ('시작', 'N'), ('하', 'X'), ('ㅂ니다', 'E'), ('.', 'S'), ('잘', 'M'), ('되', 'P'), ('겠죠', 'E'), ('?', 'S')]



```python
from konlpy.tag import Okt
t = Okt()
```


```python
print(t.nouns('한글 분석을 시작합니다. 잘 되겠죠?'))
print(t.morphs('한글 분석을 시작합니다. 잘 되겠죠?'))
print(t.pos('한글 분석을 시작합니다. 잘 되겠죠?'))
```

    ['한글', '분석', '시작']
    ['한글', '분석', '을', '시작', '합니다', '.', '잘', '되겠죠', '?']
    [('한글', 'Noun'), ('분석', 'Noun'), ('을', 'Josa'), ('시작', 'Noun'), ('합니다', 'Verb'), ('.', 'Punctuation'), ('잘', 'Verb'), ('되겠죠', 'Verb'), ('?', 'Punctuation')]


## NLTK(Natural Language Toolkit)
- NLTK : 교육용으로 개발된 자연어 처리 및 문서 분석용 파이썬 패키지  
- **말뭉치(corpus)** : 자연어 분석 작업을 위해 만든 샘플 문서 집함
    * 단순히 소설, 신문 등의 문서를 모아놓은 것도 있지만 품사, 형태소 등의 보조적 의미를 추가하고 쉬운 분석을 위해 구조적인 형태로 정리해 놓은 것을 포함.
    * NLTK의 말뭉치 자료는 설치시 제공되지 않고 download 명령어 사용자가 다운로드 받아야 함.
    
[참고 사이트](https://mangastorytelling.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%82%AC%EC%9D%B4%EC%96%B8%EC%8A%A4-%EC%8A%A4%EC%BF%A8-ml31-NLTK-%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC-%ED%8C%A8%ED%82%A4%EC%A7%80)

NLTK 설치


```python
# conda install nltk
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done
    
    # All requested packages already installed.
    
    Retrieving notices: ...working... done
    
    Note: you may need to restart the kernel to use updated packages.



```python
# conda install matplotlib
```

    Collecting package metadata (current_repodata.json): done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/anaconda3/envs/tensorflow2
    
      added / updated specs:
        - matplotlib
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        brotli-1.0.9               |       hca72f7f_7          19 KB
        brotli-bin-1.0.9           |       hca72f7f_7          17 KB
        kiwisolver-1.4.2           |  py310he9d5cce_0          61 KB
        libbrotlicommon-1.0.9      |       hca72f7f_7          69 KB
        libbrotlidec-1.0.9         |       hca72f7f_7          31 KB
        libbrotlienc-1.0.9         |       hca72f7f_7         293 KB
        libtiff-4.4.0              |       h2ef1027_0         441 KB
        libwebp-1.2.4              |       h56c3ce4_0          75 KB
        libwebp-base-1.2.4         |       hca72f7f_0         316 KB
        matplotlib-3.5.2           |  py310hecd8cb5_0           7 KB
        matplotlib-base-3.5.2      |  py310hfb0c5b7_0         5.7 MB
        pillow-9.2.0               |  py310hde71d04_1         642 KB
        zstd-1.5.2                 |       hcb37349_0         473 KB
        ------------------------------------------------------------
                                               Total:         8.1 MB
    
    The following NEW packages will be INSTALLED:
    
      brotli             pkgs/main/osx-64::brotli-1.0.9-hca72f7f_7 None
      brotli-bin         pkgs/main/osx-64::brotli-bin-1.0.9-hca72f7f_7 None
      cycler             pkgs/main/noarch::cycler-0.11.0-pyhd3eb1b0_0 None
      fonttools          pkgs/main/noarch::fonttools-4.25.0-pyhd3eb1b0_0 None
      freetype           pkgs/main/osx-64::freetype-2.11.0-hd8bbffd_0 None
      kiwisolver         pkgs/main/osx-64::kiwisolver-1.4.2-py310he9d5cce_0 None
      lcms2              pkgs/main/osx-64::lcms2-2.12-hf1fd2bf_0 None
      lerc               pkgs/main/osx-64::lerc-3.0-he9d5cce_0 None
      libbrotlicommon    pkgs/main/osx-64::libbrotlicommon-1.0.9-hca72f7f_7 None
      libbrotlidec       pkgs/main/osx-64::libbrotlidec-1.0.9-hca72f7f_7 None
      libbrotlienc       pkgs/main/osx-64::libbrotlienc-1.0.9-hca72f7f_7 None
      libdeflate         pkgs/main/osx-64::libdeflate-1.8-h9ed2024_5 None
      libtiff            pkgs/main/osx-64::libtiff-4.4.0-h2ef1027_0 None
      libwebp            pkgs/main/osx-64::libwebp-1.2.4-h56c3ce4_0 None
      libwebp-base       pkgs/main/osx-64::libwebp-base-1.2.4-hca72f7f_0 None
      lz4-c              pkgs/main/osx-64::lz4-c-1.9.3-h23ab428_1 None
      matplotlib         pkgs/main/osx-64::matplotlib-3.5.2-py310hecd8cb5_0 None
      matplotlib-base    pkgs/main/osx-64::matplotlib-base-3.5.2-py310hfb0c5b7_0 None
      munkres            pkgs/main/noarch::munkres-1.1.4-py_0 None
      pillow             pkgs/main/osx-64::pillow-9.2.0-py310hde71d04_1 None
      zstd               pkgs/main/osx-64::zstd-1.5.2-hcb37349_0 None
    
    
    
    Downloading and Extracting Packages
    kiwisolver-1.4.2     | 61 KB     | ##################################### | 100% 
    libwebp-base-1.2.4   | 316 KB    | ##################################### | 100% 
    matplotlib-base-3.5. | 5.7 MB    | ##################################### | 100% 
    brotli-1.0.9         | 19 KB     | ##################################### | 100% 
    libtiff-4.4.0        | 441 KB    | ##################################### | 100% 
    matplotlib-3.5.2     | 7 KB      | ##################################### | 100% 
    zstd-1.5.2           | 473 KB    | ##################################### | 100% 
    libbrotlidec-1.0.9   | 31 KB     | ##################################### | 100% 
    brotli-bin-1.0.9     | 17 KB     | ##################################### | 100% 
    libbrotlienc-1.0.9   | 293 KB    | ##################################### | 100% 
    libwebp-1.2.4        | 75 KB     | ##################################### | 100% 
    libbrotlicommon-1.0. | 69 KB     | ##################################### | 100% 
    pillow-9.2.0         | 642 KB    | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Retrieving notices: ...working... done
    
    Note: you may need to restart the kernel to use updated packages.



```python
import nltk
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from konlpy.corpus import kobill

from matplotlib import font_manager , rc
# path = "./ryu/Library/fonts/Arial.ttf"
# font_name= font_manager.FontProperties(fname=path).get_name()
rc('font', family='Apple Gothic')
```


```python
doc = kobill.open('1809890.txt').read()
t = Okt()
tokens = t.nouns(doc)

ko = nltk.Text(tokens, name='대한민국 국회 의안 제 1809890호')
# number of tokens
print( len(ko.tokens) )

# number of unique tokens
print( len( set(ko.tokens) ) )
```

    735
    250



```python
# frequency distribution, ko.tokens.count('육아휴직')
ko.vocab()
```




    FreqDist({'육아휴직': 38, '발생': 19, '만': 18, '이하': 18, '비용': 17, '액': 17, '경우': 16, '세': 16, '자녀': 14, '고용': 14, ...})




```python
from matplotlib import rc  ### 이 줄과
rc('font', family='AppleGothic') 			## 이 두 줄을 
plt.rcParams['axes.unicode_minus'] = False  ## 추가해줍니다. 

plt.figure(figsize=(10,8))
ko.plot(50)
plt.show()
```

![output_27_0](https://user-images.githubusercontent.com/87309905/196642567-1dcea052-014a-4d36-b13a-711dd8dd2189.png)





```python
stop_words = ['.', '(', ')', ',', '의', '자', '에', '안', '번', '호', '을', '이', '다', '및', '명', '것', '중', '안', '위', '만', '액', '제', '표']
```


```python
ko = [each_word for each_word in ko if each_word not in stop_words]
ko = nltk.Text(ko, name='대한민국 국회 의안 제 1809890호')
plt.figure(figsize=(10,8))
ko.plot(50)
plt.show()
```


![output_29_0](https://user-images.githubusercontent.com/87309905/196642489-f1ad6e42-78fd-41ba-9716-f4a74e0729b6.png)
    


---
# 17장 딥러닝을 이용한 자연어 처리

- 자연어(Natural Language)란 우리가 평소에 말하는 음성이나 텍스트를 의미.
- 자연어 처리(Natural Language Processing)는 음성이나 텍스트를 컴퓨터가 인식하고 처리

## 1. 텍스트의 토큰화
 먼저 해야할 일은 텍스트를 잘게 나누는 것입니다. 입력할 텍스트가 준비되면 이를 단어별, 문장별, 형태소별로 나눌 있는데, 이렇게 작게 나누어진 하나의 단위를 **토큰(token)**이라고 함. 예를 들어 다음과 같은 문장이 주어졌다고 가정해 보자.  
```"해보지 않으면 해낼 수 없다"```

케라스가 제공하는 text 모듈의 ```text_to_word_sequence()```함수를 사용하면 문장을 단어 단위로 나눌 수 있음. 전처리할 덱스트를 지정한 후 다음과 같이 토큰화 함.


```python
# 전처리 과정 연습

# 케라스의 텍스트 전처리와 관련한 함수 중 text_to_word_sequence 함수를 불러옵니다.
from tensorflow.keras.preprocessing.text import text_to_word_sequence

text = '해보지 않으면 해낼 수 없다'

result = text_to_word_sequence(text)

print('원문 :', text)
print('토큰화된 결과 :', result)
```

    원문 : 해보지 않으면 해낼 수 없다
    토큰화된 결과 : ['해보지', '않으면', '해낼', '수', '없다']


### [연습장]


```python
print( type(result) )
```

    <class 'list'>



```python
text = '한글 분석을 시작합니다. 잘 되겠죠?'

result = text_to_word_sequence(text)

print('원문 :', text)
print('토큰화된 결과 :', result)
```

    원문 : 한글 분석을 시작합니다. 잘 되겠죠?
    토큰화된 결과 : ['한글', '분석을', '시작합니다', '잘', '되겠죠']


이렇게 **주어진 텍스트를 단어 단위로 쪼개고 나면 이를 이용해 여러 가지를 할 수 있습니다.** 예를 들어 각 단어가 몇 번이나 중복해서 쓰였는지 알 수 있습니다. 단어의 빈도수를 알면 텍스트에서 중요한 역활을 하는 단어를 파악할 수 있겠지요. 따라서 **텍스틀 단어 단위로 쪼개는 것은 가장 많이 쓰이는 전처리 과정.** 

Bag-of-Words라는 방법이 이러한 전처리를 일컫는 말인데, '단어의 가방(bag of words)'이라는 뜻으로 같은 단어끼리 각각의 가방에 담은 후에 각 가방에 단어가 몇개 들어 있는지 세는 방법입니다. (강사-단어 출현 빈도수를 파악) 


예를 들어 다음과 같은 세 개의 문장이 있다고 합시다.
```
먼저 텍스트의 각 단어를 나누어 토큰화합니다.
텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.
토큰화한 결과는 딥러닝에서 사용할 수 있습니다.
```


케라스의 ```Tokenizer()``` 함수를 사용하면 **단어의 빈도수**를 쉽게 계산할 수 있습니다. 다음 코드는 위 제시한 세 문장의 단어를 빈도수로 다시 정리하는 코드입니다.


```python
from tensorflow.keras.preprocessing.text import Tokenizer

#전처리하려는 세 개의 문서(document)을 docs라는 리스트에 저장합니다.
docs = ['먼저 텍스트의 각 단어를 나누어 토큰화합니다. 이 문장은 어쩔라고.',
        '텍스트의 단어로 토큰화해야 딥러닝에서 인식됩니다.',
        '토큰화한 결과는 딥러닝에서 사용할 수 있습니다.'
        ]
# Tokenizer()
#  - 전처리 과정을 수행할 객체 반환
token=Tokenizer()
token.fit_on_texts(docs)  # 문서(들)을 토큰화함.

print('단어 카운트:\n', token.word_counts)   # 각 단어(token)이 전체 문서에서 몇 번 나타나는지 빈도 정보
```

    단어 카운트:
     OrderedDict([('먼저', 1), ('텍스트의', 2), ('각', 1), ('단어를', 1), ('나누어', 1), ('토큰화합니다', 1), ('이', 1), ('문장은', 1), ('어쩔라고', 1), ('단어로', 1), ('토큰화해야', 1), ('딥러닝에서', 2), ('인식됩니다', 1), ('토큰화한', 1), ('결과는', 1), ('사용할', 1), ('수', 1), ('있습니다', 1)])


```token.word_counts```에는 각 단어가 몇번 나타나는지 즉 단어 별 나타난 빈도수를 해당 단어와 같이 포함하고 있습니다.


```python
# document_count 속성에서는 총 몇 개의 문서가 들어 있는지 알 수 있음.
print('문서 카운트 : ', token.document_count )
```

    문서 카운트 :  3


## [ 실습 ]


```python
#전처리하려는 세 개의 문장을 docs라는 리스트에 저장합니다.
docs = ['난 난 꿈이 있었죠. 버려지고 찢겨 남루하여도 내 가슴 깊숙이 보물과 같이 간직했던 꿈. 혹 때론 누군가가 뜻 모를 비웃음 내 등뒤에 흘릴때도 난 참아야 했죠.',
        '참을 수 있었죠. 그 날을 위해 늘 걱정하듯 말하죠. 헛된 꿈은 독이라고 세상은 끝이 정해진 책처럼 이미 돌이킬 수 없는 현실이라고 그래요 난 난 꿈이 있어요.',
         '그 꿈을 믿어요. 나를 지켜봐요. 저 차갑게 서 있는 운명이란 벽앞에 당당히 마주칠 수 있어요.',
         '언젠가 난 그 벽을 넘고서 저 하늘을 높이 날을 수 있어요. 이 무거운 세상도 나를 묶을 순 없죠.',
        '내 삶의 끝에서 나 웃을 그 날을 함께해요.',
         '늘 걱정하듯 말하죠. 헛된 꿈은 독이라고. 세상은 끝이 정해진 책처럼 이미 돌이킬 수 없는 현실이라고.',
        '그래요. 난 난 꿈이 있어요. 그 꿈을 믿어요 나를 지켜봐요. 저 차갑게 서 있는 운명이란 벽앞에 당당히 마주칠 수 있어요.',
      '언젠가 난 그 벽을 넘고서 저 하늘을 높이 날을 수 있어요. 이 무거운 세상도 나를 묶을 순 없죠. 내 삶의 끝에서 나 웃을 그 날을 함께해요.',
      '난 난 꿈이 있어요. 그 꿈을 믿어요 나를 지켜봐요'
        ]
# Tokenizer()를 이용해 전처리 하는 과정
token = Tokenizer()
token.fit_on_texts(docs)  # 문서(들)을 토큰화 

print('단어 카운트:\n', token.word_counts)

# document_count 속성에서 총 몇 개의 문서가 들어 있는지 알 수 있음.
print('문서 카운트:\n', token.document_count )
```

    단어 카운트:
     OrderedDict([('난', 11), ('꿈이', 4), ('있었죠', 2), ('버려지고', 1), ('찢겨', 1), ('남루하여도', 1), ('내', 4), ('가슴', 1), ('깊숙이', 1), ('보물과', 1), ('같이', 1), ('간직했던', 1), ('꿈', 1), ('혹', 1), ('때론', 1), ('누군가가', 1), ('뜻', 1), ('모를', 1), ('비웃음', 1), ('등뒤에', 1), ('흘릴때도', 1), ('참아야', 1), ('했죠', 1), ('참을', 1), ('수', 7), ('그', 8), ('날을', 5), ('위해', 1), ('늘', 2), ('걱정하듯', 2), ('말하죠', 2), ('헛된', 2), ('꿈은', 2), ('독이라고', 2), ('세상은', 2), ('끝이', 2), ('정해진', 2), ('책처럼', 2), ('이미', 2), ('돌이킬', 2), ('없는', 2), ('현실이라고', 2), ('그래요', 2), ('있어요', 7), ('꿈을', 3), ('믿어요', 3), ('나를', 5), ('지켜봐요', 3), ('저', 4), ('차갑게', 2), ('서', 2), ('있는', 2), ('운명이란', 2), ('벽앞에', 2), ('당당히', 2), ('마주칠', 2), ('언젠가', 2), ('벽을', 2), ('넘고서', 2), ('하늘을', 2), ('높이', 2), ('이', 2), ('무거운', 2), ('세상도', 2), ('묶을', 2), ('순', 2), ('없죠', 2), ('삶의', 2), ('끝에서', 2), ('나', 2), ('웃을', 2), ('함께해요', 2)])
    문서 카운트:
     9


```python
token.word_counts      # 단어(token)와 해당 단어(token)가 전체 문서에 나타난 빈도
token.document_count   # 총 몇 개의 문서로 구성되어 있는지 파악
```


```python
# token.word_docs를 통해 각 단어들이 몇 개의 문서에 나오는지 세어서 출력할 수도 있습니다. 
# 출력 되는 순서는 랜덤
print('각 단어가 몇 개의 문서에 포함되어 있는가 : \n', token.word_docs)
```

    각 단어가 몇 개의 문서에 포함되어 있는가 : 
     defaultdict(<class 'int'>, {'가슴': 1, '누군가가': 1, '뜻': 1, '모를': 1, '혹': 1, '꿈': 1, '난': 6, '있었죠': 2, '간직했던': 1, '참아야': 1, '보물과': 1, '등뒤에': 1, '찢겨': 1, '버려지고': 1, '비웃음': 1, '했죠': 1, '깊숙이': 1, '남루하여도': 1, '꿈이': 4, '내': 3, '때론': 1, '흘릴때도': 1, '같이': 1, '참을': 1, '정해진': 2, '없는': 2, '있어요': 6, '늘': 2, '말하죠': 2, '그': 7, '수': 6, '날을': 4, '돌이킬': 2, '이미': 2, '위해': 1, '독이라고': 2, '꿈은': 2, '헛된': 2, '끝이': 2, '책처럼': 2, '걱정하듯': 2, '그래요': 2, '세상은': 2, '현실이라고': 2, '서': 2, '저': 4, '믿어요': 3, '지켜봐요': 3, '마주칠': 2, '벽앞에': 2, '당당히': 2, '운명이란': 2, '차갑게': 2, '나를': 5, '꿈을': 3, '있는': 2, '세상도': 2, '언젠가': 2, '하늘을': 2, '순': 2, '높이': 2, '벽을': 2, '이': 2, '넘고서': 2, '묶을': 2, '없죠': 2, '무거운': 2, '함께해요': 2, '웃을': 2, '나': 2, '끝에서': 2, '삶의': 2})



```python
# 각 단어에 매겨진 인덱스 값을 출력하려면 word_index 속성에서 확인 
print('각 단어에 매겨진 인덱스 값:\n', token.word_index)
```

    각 단어에 매겨진 인덱스 값:
     {'난': 1, '그': 2, '수': 3, '있어요': 4, '날을': 5, '나를': 6, '꿈이': 7, '내': 8, '저': 9, '꿈을': 10, '믿어요': 11, '지켜봐요': 12, '있었죠': 13, '늘': 14, '걱정하듯': 15, '말하죠': 16, '헛된': 17, '꿈은': 18, '독이라고': 19, '세상은': 20, '끝이': 21, '정해진': 22, '책처럼': 23, '이미': 24, '돌이킬': 25, '없는': 26, '현실이라고': 27, '그래요': 28, '차갑게': 29, '서': 30, '있는': 31, '운명이란': 32, '벽앞에': 33, '당당히': 34, '마주칠': 35, '언젠가': 36, '벽을': 37, '넘고서': 38, '하늘을': 39, '높이': 40, '이': 41, '무거운': 42, '세상도': 43, '묶을': 44, '순': 45, '없죠': 46, '삶의': 47, '끝에서': 48, '나': 49, '웃을': 50, '함께해요': 51, '버려지고': 52, '찢겨': 53, '남루하여도': 54, '가슴': 55, '깊숙이': 56, '보물과': 57, '같이': 58, '간직했던': 59, '꿈': 60, '혹': 61, '때론': 62, '누군가가': 63, '뜻': 64, '모를': 65, '비웃음': 66, '등뒤에': 67, '흘릴때도': 68, '참아야': 69, '했죠': 70, '참을': 71, '위해': 72}


### 1절 덱스트의 토근화에 대한 **요점 정리**
케라스가 제공하는 기능(함수)를 사용하면 text를 위한 전처리를 쉽게 할 수 있음.

---

## 2. 단어의 원-핫 인코딩
 앞서 우리는 문장을 컴퓨터가 알아 들을 수 있게 토큰화하고 단어의 비도수를 확인해 보았습니다. **(강사 - 글자를 그러니까 단어(token)을 숫자화 해야 모델 입력으로 활용할 수 있음.)** 하지만 단순히 단어의 출현 빈도만 가지고 해당 단어가 문장의 어디에서 왔는지, 각 단어의 순서는 어떠했는지 등에 관한 정보를 얻을 수 없습니다.

 단어가 문장의 다른 요소와 어떤 관계를 가지고 있는지 알아보는 방법이 필요합니다. 이러한 기법 중에서 가장 기본적인 방법인 **원-핫 인코딩**을 알아 보겠습니다. (강사-원-핫 인코딩으로 단어가 문장의 다른 요소와 어떤 관계를 가지는 알 수 있나요?) 원-핫 인코딩을 단어를 요소로하는 배열로 적용해보겠습니다. 예를 들어 다음과 같은 문장이 있습니다. 
```
'오랜동안 꿈꾸는 이는 그 꿈을 닮아간다'
```
각 단어를 모두 0으로 바꾸어 주고 원하는 단어만 1로 바꾸어 주는 것이 원-핫 인코딩이었습니다. 이를 수행하기 위해 먼저 단어 수만큼 0으로 채워진 벡터 공간으로 바꾸면 다음과 같습니다. 
```
[0 0 0 0 0 0 0]
```
<br><center>
<img src="https://drive.google.com/uc?id=1q0fo3bYP1jyyXjKew5d29axliCN4rjZB">
</center><br>
이제 각 단어가 배열 내에서 해당하는 위치를 1로 바꾸어서 벡터화 할 수 있습니다.
```
오랜동안 = [ 0 1 0 0 0 0 0]
꿈꾸는   = [ 0 0 1 0 0 0 0]
이는     = [ 0 0 0 1 0 0 0]
그       = [ 0 0 0 0 1 0 0]
꿈을     = [ 0 0 0 0 0 1 0]
닮아간다 = [ 0 0 0 0 0 0 1]
```
이러한 과정을 케라스로 구현해 보겠습니다. 먼저 토큰화 함수를 불러와 단어 단위로 토큰화하고 각 단어의 인덱스 값을 출력해 봅시다. 


```python
text = '오랫동안 꿈꾸는 이는 그 꿈을 닮아간다'
token = Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
```

    {'오랫동안': 1, '꿈꾸는': 2, '이는': 3, '그': 4, '꿈을': 5, '닮아간다': 6}


이제 각 단어를 원-핫 인코딩 방식으로 표현해 보겠습니다. 케라스에서 제공하는 ```Tokenizer```의 ```text_to_sequence()``` 함수를 사용해서 앞서 만들어진 토큰의 인텍스로만 채워진 새로운 배열을 만들어 줍니다.


```python
x = token.texts_to_sequences([text])
print(x)
```

    [[1, 2, 3, 4, 5, 6]]


이제 1\~6의 정수로 인텍스되어 있는 것을 0과 1로만 이루어진 배열로 바꾸어 주는 ```to_categorical()```함수를 사용해 워-핫 인코딩 과정을 진행합니다. 배열 맨 앞에 0이 추가됨으로 단어 수보다 1이 더 많게 인텍스 숫자를 잡아 주는 것에 유의하시기 바랍니다.


```python
from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index) + 1
x = to_categorical(x, num_classes = word_size)
print(x)
```

    [[[0. 1. 0. 0. 0. 0. 0.]
      [0. 0. 1. 0. 0. 0. 0.]
      [0. 0. 0. 1. 0. 0. 0.]
      [0. 0. 0. 0. 1. 0. 0.]
      [0. 0. 0. 0. 0. 1. 0.]
      [0. 0. 0. 0. 0. 0. 1.]]]


## 3. 단어 임베딩

```
model = Sequential()
model.add(Embedding(16, 4))
```

## 4.텍스트를 읽고 긍정, 부정 예측하기


```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding
from tensorflow.keras.utils import to_categorical
from numpy import array
```


```python
# 텍스트 리뷰 자료를 지정합니다.
docs = ["너무 재밌네요","최고예요",
        "참 잘 만든 영화예요","추천하고 싶은 영화입니다",
        "한번 더 보고싶네요","글쎄요",
        "별로예요","생각보다 지루하네요",
        "연기가 어색해요","재미없어요"]

# 긍정 리뷰는 1, 부정 리뷰는 0으로 클래스를 지정합니다.
classes = array([1,1,1,1,1,0,0,0,0,0])

# 토큰화 
token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)
```

    {'너무': 1, '재밌네요': 2, '최고예요': 3, '참': 4, '잘': 5, '만든': 6, '영화예요': 7, '추천하고': 8, '싶은': 9, '영화입니다': 10, '한번': 11, '더': 12, '보고싶네요': 13, '글쎄요': 14, '별로예요': 15, '생각보다': 16, '지루하네요': 17, '연기가': 18, '어색해요': 19, '재미없어요': 20}



```python
x = token.texts_to_sequences(docs)
print("\n리뷰 텍스트, 토큰화 결과:\n",  x)
```

    
    리뷰 텍스트, 토큰화 결과:
     [[1, 2], [3], [4, 5, 6, 7], [8, 9, 10], [11, 12, 13], [14], [15], [16, 17], [18, 19], [20]]



```python
# 패딩, 서로 다른 길이의 데이터를 4로 맞추어 줍니다.
padded_x = pad_sequences(x, 4)  
print("\n패딩 결과:\n", padded_x)
```

    
    패딩 결과:
     [[ 0  0  1  2]
     [ 0  0  0  3]
     [ 4  5  6  7]
     [ 0  8  9 10]
     [ 0 11 12 13]
     [ 0  0  0 14]
     [ 0  0  0 15]
     [ 0  0 16 17]
     [ 0  0 18 19]
     [ 0  0  0 20]]



```python
# 임베딩에 입력될 단어의 수를 지정합니다.
# 실제 단어는 20개인데 20개 데이터가 1부터 인텍싱 되기 때문에 인텍스 값은 1, 2, 3, 19, 20이다.
# 그런데 파이썬에서 테이터를 처리할 때 인텍스 0부터 고려하기 때문에 인덱스 0의 요소를 고려해서 단어가 21개인 것으로 처리하는 것으로 이해하자. 
word_size = len(token.word_index) +1
print(word_size)
```

    21



```python
# 단어 임베딩을 포함하여 딥러닝 모델을 만들고 결과를 출력합니다.
model = Sequential()

# 21차원 벡터를 8차원 벡터로 변환
# word_size: 입력될 단어 개수,
# 8: 8 차원 데이터 출력
# input_length = 4 : 한 번에 입력하는 단어 개수, [0, 0, 1, 2]
model.add(Embedding(word_size, 8, input_length=4))     # 이 예제의 경우 하나의 단어를 원핫 인코딩하면 한 단어를 나타내는 벡타는 20차원.  
                                                       # 한 단어는 첫번째 요소에 0을 추가한 것을 고려하면 21차원 벡터이다. 
                                                       # 이 벡터를 8 차원으로 바꾼다. 그리다 한번에 입력하는 단어(token)의 개수가 4개 따라서 4x8 출력
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 4, 8)              168       
                                                                     
     flatten (Flatten)           (None, 32)                0         
                                                                     
     dense (Dense)               (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 201
    Trainable params: 201
    Non-trainable params: 0
    _________________________________________________________________


    2022-10-19 11:19:00.382921: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, classes, epochs=20)
print("\n Accuracy: %.4f" % (model.evaluate(padded_x, classes)[1]))
```

    Epoch 1/20
    1/1 [==============================] - 0s 289ms/step - loss: 0.6983 - accuracy: 0.4000
    Epoch 2/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6964 - accuracy: 0.4000
    Epoch 3/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6944 - accuracy: 0.4000
    Epoch 4/20
    1/1 [==============================] - 0s 1ms/step - loss: 0.6925 - accuracy: 0.5000
    Epoch 5/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6906 - accuracy: 0.5000
    Epoch 6/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6887 - accuracy: 0.5000
    Epoch 7/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6867 - accuracy: 0.6000
    Epoch 8/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6848 - accuracy: 0.7000
    Epoch 9/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6829 - accuracy: 0.8000
    Epoch 10/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6810 - accuracy: 0.8000
    Epoch 11/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6790 - accuracy: 0.8000
    Epoch 12/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6771 - accuracy: 0.8000
    Epoch 13/20
    1/1 [==============================] - 0s 6ms/step - loss: 0.6752 - accuracy: 0.9000
    Epoch 14/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6732 - accuracy: 0.9000
    Epoch 15/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6713 - accuracy: 0.9000
    Epoch 16/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6693 - accuracy: 0.9000
    Epoch 17/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6674 - accuracy: 0.9000
    Epoch 18/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6654 - accuracy: 0.9000
    Epoch 19/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6634 - accuracy: 0.9000
    Epoch 20/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6614 - accuracy: 0.9000
    1/1 [==============================] - 0s 74ms/step - loss: 0.6594 - accuracy: 0.9000
    
     Accuracy: 0.9000



```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_x, classes, epochs=20)
print("\n Accuracy: %.4f" % (model.evaluate(padded_x, classes)[1]))
```

    Epoch 1/20
    1/1 [==============================] - 0s 217ms/step - loss: 0.6594 - accuracy: 0.9000
    Epoch 2/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6573 - accuracy: 0.9000
    Epoch 3/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6553 - accuracy: 0.9000
    Epoch 4/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6533 - accuracy: 0.9000
    Epoch 5/20
    1/1 [==============================] - 0s 4ms/step - loss: 0.6512 - accuracy: 0.9000
    Epoch 6/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6491 - accuracy: 0.9000
    Epoch 7/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6470 - accuracy: 0.9000
    Epoch 8/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6450 - accuracy: 0.9000
    Epoch 9/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6428 - accuracy: 0.9000
    Epoch 10/20
    1/1 [==============================] - 0s 4ms/step - loss: 0.6407 - accuracy: 0.9000
    Epoch 11/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6386 - accuracy: 0.9000
    Epoch 12/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6365 - accuracy: 0.9000
    Epoch 13/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6343 - accuracy: 0.9000
    Epoch 14/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6321 - accuracy: 0.9000
    Epoch 15/20
    1/1 [==============================] - 0s 3ms/step - loss: 0.6299 - accuracy: 0.9000
    Epoch 16/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6277 - accuracy: 0.9000
    Epoch 17/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6255 - accuracy: 0.9000
    Epoch 18/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6233 - accuracy: 0.9000
    Epoch 19/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6211 - accuracy: 0.9000
    Epoch 20/20
    1/1 [==============================] - 0s 2ms/step - loss: 0.6188 - accuracy: 0.9000
    1/1 [==============================] - 0s 57ms/step - loss: 0.6165 - accuracy: 0.9000
    
     Accuracy: 0.9000

