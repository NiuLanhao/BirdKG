from py2neo import Graph, NodeMatcher, RelationshipMatcher

uri = "bolt://localhost:7687"  # 使用 "bolt" 作为默认协议和端口
username = "neo4j"
password = "nlh20020922"


class Neo4j:
    def __init__(self):
        self.graph = None
        print("create neo4j class ...")
        self.uri = uri
        self.username = username
        self.password = password

    def connectDB(self):
        self.graph = Graph(self.uri, auth=(self.username, self.password))

    def closeDB(self):
        self.graph = None

    def print_limited_results(self, results, limit):
        for i, result in enumerate(results[:limit]):
            print(f"{i + 1}. {result}")

    def matchItembyTitle(self, value):
        query = (
            "MATCH (n {name: $value})"
            "RETURN n"
        )
        answer = self.graph.run(query, value=value).data()
        # print("matchItembyTitle Result:")
        # self.print_limited_results(answer, limit=3)
        return answer

    # 实体详情查询
    def matchHudongItembyTitle(self, value):
        query = (
            "MATCH (n {name: $value})"
            "RETURN n"
        )
        try:
            answer = self.graph.run(query, value=value).data()
        except:
            print(query)
        # print("matchHudongItembyTitle Result:")
        # self.print_limited_results(answer, limit=3)
        return answer

    def getEntityRelationbyEntity(self, value):
        query = (
            "MATCH (entity1) - [rel] -> (entity2) "
            "WHERE entity1.name = $name "
            "RETURN rel, entity2"
        )
        answer = self.graph.run(query, name=value).data()
        # print("getEntityRelationbyEntity Result:")
        # self.print_limited_results(answer, limit=3)
        return answer

    def findRelationByEntity(self, entity1):
        query = (
            "MATCH (n1 {name: $entity1})- [rel] -> (n2) "
            "RETURN n1, rel, n2"
        )
        result = self.graph.run(query, entity1=entity1).data()
        # print("findRelationByEntity Result:")
        # self.print_limited_results(result, limit=3)
        return result

    def findRelationByEntity2(self, entity1):
        query = (
            "MATCH (n1)- [rel] -> (n2 {name: $entity1}) "
            "RETURN n1, rel, n2"
        )
        result = self.graph.run(query, entity1=entity1).data()
        # print("findRelationByEntity2 Result:")
        # self.print_limited_results(result, limit=3)
        return result

    def findOtherEntities(self, entity, relation):
        query = (
            "MATCH (n1 {name: $entity})- [rel {type: $relation}] -> (n2) "
            "RETURN n1, rel, n2"
        )
        result = self.graph.run(query, entity=entity, relation=relation).data()
        # print("findOtherEntities Result:")
        # self.print_limited_results(result, limit=3)
        return result

    def findOtherEntities2(self, entity, relation):
        query = (
            "MATCH (n1)- [rel {type: $relation}] -> (n2 {name: $entity}) "
            "RETURN n1, rel, n2"
        )
        result = self.graph.run(query, entity=entity, relation=relation).data()
        # print("findOtherEntities2 Result:")
        # self.print_limited_results(result, limit=3)
        return result

    def findRelationByEntities(self, entity1, entity2):
        query = (
            "MATCH (p1 {name: $entity1})-[rel:RELATION*]-(p2 {name: $entity2}) "
            "RETURN rel"
        )
        result = self.graph.run(query, entity1=entity1, entity2=entity2).data()
        # print("findRelationByEntities Result:")
        # self.print_limited_results(result, limit=3)
        return result

    def findEntityRelation(self, entity1, relation, entity2):
        query = (
            "MATCH (n1 {name: $entity1})-[rel:RELATION {type: $relation}]->(n2 {name: $entity2}) "
            "RETURN n1, rel, n2"
        )
        result = self.graph.run(query, entity1=entity1, relation=relation, entity2=entity2).data()
        # print("findEntityRelation Result:")
        # self.print_limited_results(result, limit=3)
        return result
