import sys
import re
import collections
import marshal

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
      
    f.close()
    return (queries, features)

def getRawCounts(queries, features, query, result):
  queryTerms = query.rsplit()
  
  urlSeparatedNonAlphaNum = re.sub(r'[^A-Za-z9-9]', ' ', result).split()
  rawURLCounts = collections.defaultdict(lambda: 0)
  for queryTerm in queryTerms:
    queryTerm = queryTerm.lower()
    count = 0
    for urlTerm in urlSeparatedNonAlphaNum:
      if urlTerm.lower() == queryTerm:
        count += 1
    rawURLCounts[queryTerm] = rawURLCounts[queryTerm] + count

  rawTitleCounts = collections.defaultdict(lambda: 0)
  if 'title' in features[query][result]:
    title = features[query][result]['title'].split()
    for queryTerm in queryTerms:
      queryTerm = queryTerm.lower()
      count = 0
      for titleTerm in title:
        if titleTerm.lower() == queryTerm:
          count += 1
      rawTitleCounts[queryTerm] = rawTitleCounts[queryTerm] + count

  rawHeaderCounts = collections.defaultdict(lambda: 0)
  if 'header' in features[query][result]:
    headerTerms = ' '.join(features[query][result]['header']).split()
    for queryTerm in queryTerms:
      queryTerm = queryTerm.lower()
      count = 0
      for headerTerm in headerTerms:
        if headerTerm.lower() == queryTerm:
          count += 1
      rawHeaderCounts[queryTerm] = rawHeaderCounts[queryTerm] + count

  rawBodyHitCounts = collections.defaultdict(lambda: 0)
  if 'body_hits' in features[query][result]:
    bodyHits = features[query][result]['body_hits']
    for queryTerm in queryTerms:
      queryTerm = queryTerm.lower()
      if queryTerm in bodyHits:
        rawBodyHitCounts[queryTerm] = rawBodyHitCounts[queryTerm] +\
          len(bodyHits[queryTerm])
  
  rawAnchorHitCounts = collections.defaultdict(lambda: 0)
  if 'anchors' in features[query][result]:
    fullAnchorText = []
    for (anchorText, nAnchors) in\
      features[query][result]['anchors'].iteritems():
      anchorTextTerms = anchorText.split()
      for term in anchorTextTerms:
        fullAnchorText.append((term, nAnchors))
    for queryTerm in queryTerms:
      queryTerm = queryTerm.lower()
      count = 0
      for anchor in fullAnchorText:
        if anchor[0].lower() == queryTerm:
          count += anchor[1]
      rawAnchorHitCounts[queryTerm] = rawAnchorHitCounts[queryTerm] + count

  return [rawURLCounts, rawTitleCounts, rawHeaderCounts, rawBodyHitCounts,
          rawAnchorHitCounts]

#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

